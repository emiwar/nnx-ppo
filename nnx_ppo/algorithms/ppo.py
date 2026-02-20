from typing import Any, Optional
from collections.abc import Callable
import dataclasses

import numpy as np

from flax import nnx
import jax
import jax.numpy as jp
from jax.experimental import checkify
from jaxtyping import Array, Float, Bool, PRNGKeyArray, ScalarLike, Integer
import optax

from nnx_ppo.networks.types import PPONetwork, ModuleState
from nnx_ppo.algorithms import rollout
from nnx_ppo.algorithms.types import TrainingState, LoggingLevel, RLEnv, EnvState
from nnx_ppo.algorithms.config import (
    TrainConfig,
    PPOConfig,
    EvalConfig,
    VideoConfig,
    VideoData,
    TrainResult,
)
from nnx_ppo.algorithms.metrics import compute_metrics, log_weight_stats

def default_config() -> TrainConfig:
    """Return default training configuration."""
    return TrainConfig()


def _should_run(steps: int, last_step: int, every_steps: int) -> bool:
    """Check if we should run an action at this step count."""
    if every_steps <= 0:
        return False
    return (steps // every_steps) > (last_step // every_steps)


def train_ppo(
    env: RLEnv,
    networks: PPONetwork,
    config: Optional[TrainConfig] = None,
    *,
    total_steps: Optional[int] = None,
    seed: Optional[int] = None,
    log_fn: Optional[Callable[[dict[str, Any], int], None]] = None,
    video_fn: Optional[Callable[[VideoData], None]] = None,
    eval_env: Optional[RLEnv] = None,
    initial_state: Optional[TrainingState] = None,
) -> TrainResult:
    """Train a PPO agent.

    Args:
        env: Training environment (MjxEnv).
        networks: PPO network (actor-critic).
        config: Training configuration. If None, uses default_config().
        total_steps: Override config.ppo.total_steps (convenience parameter).
        seed: Override config.seed (convenience parameter).
        log_fn: Called with (metrics_dict, step) after each PPO step.
                If None, no logging is performed.
        video_fn: Called with VideoData after rendering eval episodes.
                  If None, no videos are recorded even if config.video.enabled.
        eval_env: Environment for evaluation rollouts. If None, uses env.
        initial_state: Resume training from an existing TrainingState.
                       If None, creates a new TrainingState.

    Returns:
        TrainResult containing final TrainingState, metrics, and eval history.
    """
    # Setup config with overrides
    if config is None:
        config = default_config()
    if total_steps is not None:
        config = dataclasses.replace(
            config, ppo=dataclasses.replace(config.ppo, total_steps=total_steps)
        )
    if seed is not None:
        config = dataclasses.replace(config, seed=seed)

    # Setup eval_env
    if eval_env is None:
        eval_env = env

    # Initialize or resume training state
    if initial_state is None:
        training_state = new_training_state(
            env,
            networks,
            config.ppo.n_envs,
            config.seed,
            config.ppo.learning_rate,
            config.ppo.gradient_clipping,
            config.ppo.weight_decay,
        )
    else:
        training_state = initial_state

    # JIT compile functions
    ppo_step_jit = nnx.jit(ppo_step, static_argnums=(0, 2, 3, 7, 8, 9, 10, 11))
    eval_rollout_jit = nnx.jit(rollout.eval_rollout, static_argnums=(0, 2, 3, 5))
    eval_rollout_render_jit = nnx.jit(
        rollout.eval_rollout_for_render_scan, static_argnums=(0, 2)
    )

    # Training loop state
    eval_history: list[dict[str, Any]] = []
    last_eval_step = -config.eval.every_steps  # Ensure eval at step 0
    last_video_step = -config.video.every_steps  # Ensure video at step 0
    metrics: dict[str, Any] = {}
    n_iterations = 0

    # Helper for running eval
    def run_eval(steps: int) -> dict[str, Any]:
        networks.eval()
        eval_metrics = eval_rollout_jit(
            eval_env,
            networks,
            config.eval.n_envs,
            config.eval.max_episode_length,
            jax.random.key(config.seed),
            config.eval.logging_percentiles,
        )
        networks.train()
        return dict(eval_metrics)

    # Helper for running video
    def run_video(steps: int, iteration: int) -> None:
        if video_fn is None or not hasattr(eval_env, "render"):
            return
        networks.eval()
        render_key = jax.random.fold_in(jax.random.key(config.seed), iteration)
        stacked_states, final_state, episode_reward = eval_rollout_render_jit(
            eval_env, networks, config.video.episode_length, render_key
        )
        trajectory = rollout.unstack_trajectory(
            stacked_states, final_state, config.video.episode_length
        )
        frames = eval_env.render(trajectory, **config.video.render_kwargs)
        video_data = VideoData(
            frames=np.stack(frames),
            step=steps,
            episode_reward=float(episode_reward),
            episode_length=config.video.episode_length,
        )
        video_fn(video_data)
        networks.train()

    # Initial eval/video at step 0
    steps = int(training_state.steps_taken)
    if config.eval.enabled:
        eval_metrics = run_eval(steps)
        metrics.update(eval_metrics)
        eval_history.append({"step": steps, **eval_metrics})
        last_eval_step = steps
    if config.video.enabled:
        run_video(steps, n_iterations)
        last_video_step = steps
    if log_fn is not None and metrics:
        log_fn(metrics, steps)

    # Main training loop
    while int(training_state.steps_taken) < config.ppo.total_steps:
        # PPO step
        training_state, metrics = ppo_step_jit(
            env,
            training_state,
            config.ppo.n_envs,
            config.ppo.rollout_length,
            config.ppo.gae_lambda,
            config.ppo.discounting_factor,
            config.ppo.clip_range,
            config.ppo.normalize_advantages,
            config.ppo.n_epochs,
            config.ppo.n_minibatches,
            config.ppo.logging_level,
            config.ppo.logging_percentiles,
        )
        n_iterations += 1
        steps = int(training_state.steps_taken)

        # Eval rollout
        if config.eval.enabled and _should_run(
            steps, last_eval_step, config.eval.every_steps
        ):
            eval_metrics = run_eval(steps)
            metrics.update(eval_metrics)
            eval_history.append({"step": steps, **eval_metrics})
            last_eval_step = steps

        # Video recording
        if config.video.enabled and _should_run(
            steps, last_video_step, config.video.every_steps
        ):
            run_video(steps, n_iterations)
            last_video_step = steps

        # Logging
        if log_fn is not None:
            log_fn(metrics, steps)

    # Return result
    return TrainResult(
        training_state=training_state,
        final_metrics=metrics,
        eval_history=eval_history,
        total_steps=int(training_state.steps_taken),
        total_iterations=n_iterations,
    )


def ppo_step(
    env: RLEnv,
    training_state: TrainingState,
    n_envs: int,
    rollout_length: int,
    gae_lambda: ScalarLike,
    discounting_factor: ScalarLike,
    clip_range: ScalarLike,
    normalize_advantages: bool,
    n_epochs: int,
    n_minibatches: int,
    logging_level: LoggingLevel = LoggingLevel.LOSSES,
    logging_percentiles: Optional[tuple[int, ...]] = None,
) -> tuple[TrainingState, dict[str, Any]]:

    reset_key, new_key = jax.random.split(training_state.rng_key)
    next_net_state, next_env_state, rollout_data = rollout.unroll_env(
        env,
        training_state.env_states,
        training_state.networks,
        training_state.network_states,
        rollout_length,
        reset_key,
    )

    grad_fn = nnx.grad(ppo_loss, has_aux=True)

    # Pre-compute all minibatch indices for all epochs
    total_iterations = n_epochs * n_minibatches
    minibatch_size = n_envs // n_minibatches

    def get_epoch_indices(epoch_idx):
        shuffle_key = jax.random.fold_in(new_key, epoch_idx)
        perm = jax.random.permutation(shuffle_key, n_envs)
        return perm.reshape(n_minibatches, minibatch_size)

    # Shape: (n_epochs, n_minibatches, minibatch_size) -> (n_epochs * n_minibatches, minibatch_size)
    all_indices = jax.vmap(get_epoch_indices)(jp.arange(n_epochs))
    all_indices = all_indices.reshape(total_iterations, minibatch_size)

    def update_step(networks, optimizer, inds):
        minibatch_data = jax.tree.map(lambda x: x[:, inds], rollout_data)
        net_state_subset = jax.tree.map(
            lambda x: x[inds], training_state.network_states
        )
        grads, loss_metrics = grad_fn(
            networks=networks,
            network_state=net_state_subset,
            rollout_data=minibatch_data,
            clip_range=clip_range,
            normalize_advantages=normalize_advantages,
            discounting_factor=discounting_factor,
            gae_lambda=gae_lambda,
            logging_level=logging_level,
            logging_percentiles=logging_percentiles,
        )
        if LoggingLevel.GRAD_NORM in logging_level:
            grad_norm = jp.sqrt(sum(jp.sum(g**2) for g in jax.tree.leaves(grads)))
            loss_metrics["grad_norm"] = grad_norm
        optimizer.update(networks, grads)
        return loss_metrics

    scan_update = nnx.scan(
        update_step,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.StateAxes({...: nnx.Carry}), 0),
        out_axes=0,
        length=total_iterations,
    )

    loss_metrics = scan_update(
        training_state.networks, training_state.optimizer, all_indices
    )
    total_steps = training_state.steps_taken + rollout_length * n_envs
    metrics = compute_metrics(
        loss_metrics, rollout_data, logging_level, logging_percentiles
    )
    metrics["total_steps"] = total_steps
    if LoggingLevel.WEIGHTS in logging_level:
        log_weight_stats(metrics, training_state.networks, logging_percentiles)
    training_state.networks.update_statistics(rollout_data, total_steps)

    # Now that all updates are done, we can replace all the network (and environment)
    # states in training state. Note that this would have been incorrect to update
    # earlier (see note above).
    training_state = training_state.replace(
        network_states=next_net_state,
        env_states=next_env_state,
        rng_key=new_key,
        steps_taken=total_steps,
    )

    return training_state, metrics


def gae(
    rewards: Float[Array, "time batch"],
    values: Float[Array, "time_plus_1 batch"],
    done: Bool[Array, "time batch"],
    truncation: Bool[Array, "time batch"],
    lambda_: ScalarLike,
    gamma: ScalarLike,
) -> Float[Array, "time batch"]:
    assert values.shape == (rewards.shape[0] + 1, rewards.shape[1])

    def inner_step(next_advantage, reward, old_value, next_value, done, truncated):
        next_value = jp.where(done, 0.0, next_value)
        new_value = reward + gamma * next_value
        advantage = new_value - old_value
        advantage = jp.where(truncated, 0.0, advantage)
        gae_advantage = advantage + (1 - done) * gamma * lambda_ * next_advantage
        return gae_advantage, gae_advantage

    time_scan = nnx.scan(
        inner_step,
        in_axes=(nnx.Carry, 0, 0, 0, 0, 0),
        out_axes=(nnx.Carry, 0),
        length=rewards.shape[0],
        reverse=True,
    )
    _, advantages = time_scan(
        next_advantage=jp.zeros(rewards.shape[1]),
        reward=rewards,
        old_value=values[:-1, :],
        next_value=values[1:, :],
        done=done,
        truncated=truncation,
    )
    return advantages


def ppo_loss(
    networks: PPONetwork,
    network_state: Any,
    rollout_data: rollout.Transition,
    clip_range: ScalarLike,
    normalize_advantages: bool,
    discounting_factor: ScalarLike,
    gae_lambda: ScalarLike,
    logging_level: LoggingLevel,
    logging_percentiles: Optional[tuple[int, ...]] = None,
) -> tuple[Float[Array, ""], dict[str, Any]]:
    rollout_data = jax.lax.stop_gradient(rollout_data)

    @jax.vmap
    def reset_net_state(done, state):
        return jax.lax.cond(done, networks.reset_state, lambda x: x, state)

    def step_network(networks: PPONetwork, net_state, obs, done, raw_action):
        net_state, network_output = networks(net_state, obs, raw_action)
        net_state = reset_net_state(done, net_state)
        return net_state, network_output

    time_scan = nnx.scan(
        step_network,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0, 0, 0),
        out_axes=(nnx.Carry, 0),
    )
    next_net_state_again, network_output = time_scan(
        networks,
        network_state,
        rollout_data.obs,
        rollout_data.done,
        rollout_data.network_output.raw_actions,
    )

    last_obs = jax.tree.map(lambda x: x[-1], rollout_data.next_obs)
    _, network_output_last = networks(next_net_state_again, last_obs)
    last_value = jax.lax.stop_gradient(network_output_last.value_estimates)
    last_value = last_value.reshape((1, last_value.shape[0]))
    values_excl_last = network_output.value_estimates
    values_incl_last = jp.concatenate((values_excl_last, last_value), axis=0)
    advantages = gae(
        rewards=rollout_data.rewards,
        values=values_incl_last,
        done=rollout_data.done,
        truncation=rollout_data.truncated,
        lambda_=gae_lambda,
        gamma=discounting_factor,
    )
    advantages = jax.lax.stop_gradient(advantages)
    target_values = jax.lax.stop_gradient(values_excl_last + advantages)

    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = jax.lax.stop_gradient(advantages)

    old_loglikelihoods = jax.lax.stop_gradient(
        rollout_data.network_output.loglikelihoods
    )
    likelihood_ratios = jp.exp(network_output.loglikelihoods - old_loglikelihoods)
    loss_cand1 = likelihood_ratios * advantages
    loss_cand2 = jp.clip(likelihood_ratios, 1 - clip_range, 1 + clip_range) * advantages

    # Note that it's the network's responsiblity to add entropy loss as one particular
    # instance of a regularization loss.
    actor_loss = -jp.mean(jp.minimum(loss_cand1, loss_cand2))
    critic_loss = 0.5 * jp.mean((network_output.value_estimates - target_values) ** 2)
    regularization_loss = jp.mean(network_output.regularization_loss)

    loss_metrics = dict()
    if LoggingLevel.LOSSES in logging_level:
        loss_metrics["losses/actor"] = actor_loss
        loss_metrics["losses/critic"] = critic_loss
        loss_metrics["losses/regularization"] = regularization_loss
    if LoggingLevel.ACTOR_EXTRA in logging_level:
        loss_metrics["correlations/ll_advantage"] = jp.corrcoef(
            rollout_data.network_output.loglikelihoods.flatten(), advantages.flatten()
        )[0, 1]
        loss_metrics["losses/likelihood_ratios"] = likelihood_ratios
        loss_metrics["losses/likelihood_ratios_mean"] = jp.mean(likelihood_ratios)
        loss_metrics["losses/clipping_fraction"] = jp.mean(
            jp.logical_or(
                likelihood_ratios < 1 - clip_range, likelihood_ratios > 1 + clip_range
            )
        )
        loss_metrics["losses/new_loglikelihoods"] = network_output.loglikelihoods
        loss_metrics["losses/loglikelihood_diff"] = (
            network_output.loglikelihoods - old_loglikelihoods
        )
        loss_metrics["losses/new_mu"] = network_output.metrics["action_sampler"]["mu"]
        loss_metrics["losses/new_sigma"] = network_output.metrics["action_sampler"][
            "sigma"
        ]
        loss_metrics["losses/mu_diff"] = (
            network_output.metrics["action_sampler"]["mu"]
            - rollout_data.network_output.metrics["action_sampler"]["mu"]
        )
        loss_metrics["losses/sigma_diff"] = (
            network_output.metrics["action_sampler"]["sigma"]
            - rollout_data.network_output.metrics["action_sampler"]["sigma"]
        )
        loss_metrics["losses/sigma_ratio"] = (
            network_output.metrics["action_sampler"]["sigma"]
            / rollout_data.network_output.metrics["action_sampler"]["sigma"]
        )
    if LoggingLevel.CRITIC_EXTRA in logging_level:
        loss_metrics["losses/predicted_value"] = values_excl_last
        loss_metrics["losses/advantages"] = advantages
        loss_metrics["losses/advantages_NaN"] = 1.0 - jp.isfinite(advantages).mean()
        loss_metrics["losses/critic_R^2"] = 1.0 - 2 * critic_loss / (
            jp.var(target_values) + 1e-8
        )

    total_loss = actor_loss + critic_loss + regularization_loss

    # Sometimes, for some inexplicable reason, the network produces garbage outputs
    # during this function, but not during earlier rollouts. So a heuristic is that
    # if the _mean_ likelihood ratio is out of clipping bounds, the minibatch is bad
    # and we just ignore it by setting the loss to 0.0.
    # total_loss *= jp.median(likelihood_ratios) > (1-clip_range)
    # total_loss *= jp.median(likelihood_ratios) < (1+clip_range)

    return total_loss, loss_metrics


def new_training_state(
    env: RLEnv,
    networks: PPONetwork,
    n_envs: int,
    seed: int | Integer[Array, ""],
    learning_rate: float = 1e-4,
    gradient_clipping: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> TrainingState:
    # Setup keys
    key = jax.random.key(seed)
    key, training_key = jax.random.split(key)

    # Setup environment states
    env_init_keys = jax.random.split(key, n_envs)
    env_states = nnx.vmap(env.reset)(env_init_keys)

    # Setup network states
    network_states = networks.initialize_state(n_envs)

    # Setup optimizer
    optimizer_chain_links = []
    if gradient_clipping is not None:
        optimizer_chain_links.append(optax.clip_by_global_norm(gradient_clipping))
    if weight_decay is None:
        optimizer_chain_links.append(optax.adam(learning_rate=learning_rate))
    elif isinstance(weight_decay, bool) and weight_decay:
        # Optax default decay
        optimizer_chain_links.append(optax.adamw(learning_rate=learning_rate))
    else:
        optimizer_chain_links.append(
            optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        )
    optimizer = nnx.Optimizer(
        networks, optax.chain(*optimizer_chain_links), wrt=nnx.Param
    )
    return TrainingState(
        networks, network_states, env_states, optimizer, training_key, jp.array(0.0)
    )


def checkify_tree_equals(A, B, msg: str):
    jax.tree.map(lambda a, b: checkify.check(jp.all(a == b), msg), A, B)
