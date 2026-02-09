from typing import Any, Optional, Tuple, Dict, Union, Mapping
import enum
import copy
import warnings

from ml_collections import config_dict
import mujoco_playground
import flax.struct
from flax import nnx
import jax
import jax.numpy as jp
from jax.experimental import checkify
import optax

from nnx_ppo.networks.types import PPONetwork, PPONetworkOutput
from nnx_ppo.algorithms import rollout

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        n_envs = 256,
        rollout_length = 20,
        n_steps = 256*20*100,
        gae_lambda = 0.95,
        discounting_factor = 0.9,
        clip_range = 0.2,
        learning_rate = 1e-4,
        normalize_advantages = True,
        n_epochs = 4,
        n_minibatches = 4,
    )

@flax.struct.dataclass
class TrainingState:
    networks: PPONetwork
    network_states: Any
    env_states: mujoco_playground.State
    optimizer: nnx.Optimizer
    rng_key: jax.Array
    steps_taken: jax.Array

class LoggingLevel(enum.Flag):
    LOSSES = enum.auto()
    CRITIC_EXTRA = enum.auto()
    ACTOR_EXTRA = enum.auto()
    TRAIN_ROLLOUT_STATS = enum.auto()
    TRAINING_ENV_METRICS = enum.auto()
    GRAD_NORM = enum.auto()
    BASIC = LOSSES
    ALL = LOSSES | ACTOR_EXTRA | CRITIC_EXTRA | TRAIN_ROLLOUT_STATS | TRAINING_ENV_METRICS
    NONE = 0

def train_ppo(env: mujoco_playground.MjxEnv,
              networks: PPONetwork,
              config: config_dict.ConfigDict = default_config(),
              seed: int = 17,
              logging_level: LoggingLevel = LoggingLevel.BASIC):
    training_state = new_training_state(env, networks, config.n_envs, seed)
    ppo_step_jit = nnx.jit(ppo_step, static_argnums=(0, 2, 3, 7, 8, 9, 10))
    metrics = None
    while training_state.steps_taken < config.n_steps:
        training_state, metrics = ppo_step_jit(
            env, training_state, 
            config.n_envs, config.rollout_length,
            config.gae_lambda, config.discounting_factor,
            config.clip_range, config.normalize_advantages,
            config.n_epochs, config.n_minibatches, logging_level
        )
    return training_state, metrics

def ppo_step(env: mujoco_playground.MjxEnv,
             training_state: TrainingState,
             n_envs: int,
             rollout_length: int,
             gae_lambda: float,
             discounting_factor: float,
             clip_range: float,
             normalize_advantages: bool,
             n_epochs: int,
             n_minibatches: int,
             logging_level: LoggingLevel = LoggingLevel.LOSSES,
             logging_percentiles: Optional[Tuple] = None) -> Tuple[TrainingState, Dict]:
    
    reset_key, new_key = jax.random.split(training_state.rng_key)
    next_net_state, next_env_state, rollout_data = rollout.unroll_env(
        env,
        training_state.env_states,
        training_state.networks,
        training_state.network_states,
        rollout_length,
        reset_key
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
        net_state_subset = jax.tree.map(lambda x: x[inds], training_state.network_states)
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
        length=total_iterations
    )

    all_metrics = scan_update(training_state.networks, training_state.optimizer, all_indices)
    # Take the last metrics from the scan
    metrics = jax.tree.map(lambda x: x[-1], all_metrics)

    _log_metrics(metrics, rollout_data, logging_level, logging_percentiles)

    # Now that all updates are done, we can replace all the network (and environment)
    # states in training state. Note that this would have been incorrect to update
    # earlier (see note above).
    training_state = training_state.replace(
        network_states = next_net_state,
        env_states = next_env_state,
        rng_key = new_key,
        steps_taken = training_state.steps_taken + rollout_length * n_envs,
    )
    metrics["total_steps"] = training_state.steps_taken
    return training_state, metrics

def _log_metrics(metrics: Dict[str, jax.Array],
                 rollout_data: rollout.Transition,
                 logging_level: LoggingLevel,
                 percentile_levels: Optional[Tuple] = None):
    if LoggingLevel.TRAINING_ENV_METRICS in logging_level:
        for k, v in rollout_data.metrics.items():
            _log_metric(metrics, k, v, percentile_levels)
    if LoggingLevel.TRAIN_ROLLOUT_STATS in logging_level:
        _log_metric(metrics, "rollout_batch/reward", rollout_data.rewards, percentile_levels)
        _log_metric(metrics, "rollout_batch/action", rollout_data.network_output.actions, percentile_levels)
        metrics["rollout_batch/done_rate"] = rollout_data.done.mean()
        metrics["rollout_batch/truncation_rate"] = rollout_data.truncated.mean()
    if LoggingLevel.ACTOR_EXTRA in logging_level:
        _log_metric(metrics, "loglikelihood", rollout_data.network_output.loglikelihoods, percentile_levels)
        if rollout_data.network_output.actions.shape[-1] == 1:
            metrics["correlations/action_ll"] = jp.corrcoef(rollout_data.network_output.loglikelihoods.flatten(),
                                                    rollout_data.network_output.actions.flatten())[0, 1]
    if LoggingLevel.CRITIC_EXTRA in logging_level:
        _log_metric(metrics, "losses/predicted_value", rollout_data.network_output.value_estimates, percentile_levels)

def _log_metric(metrics: Dict[str, jax.Array], name: str, x: Union[Mapping, jax.Array], percentile_levels: Optional[Tuple] = None):
    if isinstance(x, Mapping):
        for k, v in x.items():
            _log_metric(metrics, f"{name}/{k}", v, percentile_levels)
        return
    if name.startswith("env/termination"): #These are boolean, but casted to float earlier
        metrics[name] = jp.mean(x)
    elif percentile_levels is None or len(percentile_levels) == 0:
        metrics[f"{name}/mean"] = jp.mean(x)
        metrics[f"{name}/std"] = jp.std(x)
    else:
        percentiles = jp.percentile(x, jp.array(percentile_levels))
        for (pl, p) in zip(percentile_levels, percentiles):
            metrics[f"{name}/p{int(pl)}"] = p


def gae(rewards, values, done, truncation, lambda_: float, gamma: float):
    assert values.shape == (rewards.shape[0]+1, rewards.shape[1])
    assert rewards.shape == done.shape
    assert truncation.shape == done.shape
    def inner_step(next_advantage, reward, old_value, next_value, done, truncated):
        next_value *= (1 - done)
        new_value = reward + gamma * next_value
        advantage = new_value - old_value
        advantage *= (1 - truncated)
        gae_advantage = advantage + (1 - done) * gamma * lambda_ * next_advantage
        return gae_advantage, gae_advantage
    time_scan = nnx.scan(inner_step,
        in_axes=(nnx.Carry, 0, 0, 0, 0, 0), out_axes=(nnx.Carry, 0),
        length=rewards.shape[0], reverse=True)
    _, advantages = time_scan(next_advantage = jp.zeros(rewards.shape[1]),
                              reward=rewards,
                              old_value=values[:-1, :],
                              next_value=values[1:, :],
                              done=done,
                              truncated=truncation)
    return advantages

def ppo_loss(networks: PPONetwork,
             network_state,
             rollout_data: rollout.Transition,
             clip_range: float,
             normalize_advantages: bool,
             discounting_factor: float,
             gae_lambda: float,
             logging_level: LoggingLevel,
             logging_percentiles: Optional[Tuple] = None):
    rollout_data = jax.lax.stop_gradient(rollout_data)
    metrics = dict()

    @jax.vmap
    def reset_net_state(done, state):
        return jax.lax.cond(done, networks.initialize_state, lambda _: state, done.shape)
    
    def step_network(networks: PPONetwork, net_state, obs, done, raw_action):
        net_state, network_output = networks(net_state, obs, raw_action, gradient_mode=True)
        net_state = reset_net_state(done, net_state)
        return net_state, network_output

    time_scan = nnx.scan(step_network, in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0, 0, 0), out_axes=(nnx.Carry, 0))
    next_net_state_again, network_output = time_scan(networks, network_state, rollout_data.obs, rollout_data.done, rollout_data.network_output.raw_actions)
    
    if network_output.value_estimates.ndim == 3:
        assert network_output.value_estimates.shape[2] == 1
        network_output = network_output.replace(
            value_estimates = network_output.value_estimates[:, :, 0]
        )

    #if LoggingLevel.ASSERTS in logging_level:
    #    metrics["asserts/actions_max_diff"] = jp.max(jp.abs(network_output.actions - rollout_data.network_output.actions))
    #    metrics["asserts/likelihoods_max_diff"] = jp.max(jp.abs(network_output.loglikelihoods - rollout_data.network_output.loglikelihoods))
    #    metrics["asserts/critic_values_max_diff"] = jp.max(jp.abs(network_output.value_estimates - rollout_data.network_output.value_estimates))
    #    metrics["asserts/net_state_identical"] = jax.tree.reduce(jp.logical_and, jax.tree.map(jp.allclose, next_net_state_again, next_net_state), jp.array(True)).astype(int)
    # We need the value of the final observation
    last_obs = jax.tree.map(lambda x: x[-1], rollout_data.next_obs)
    _, network_output_last = networks(next_net_state_again, last_obs, gradient_mode=True)
    last_value = jax.lax.stop_gradient(network_output_last.value_estimates)
    assert last_value.shape[0] == rollout_data.rewards.shape[1]
    last_value = last_value.reshape((1, last_value.shape[0]))
    values_excl_last = network_output.value_estimates
    values_incl_last = jp.concatenate((values_excl_last, last_value), axis=0)
    advantages = gae(rewards=rollout_data.rewards,
                     values=values_incl_last,
                     done=rollout_data.done,
                     truncation=rollout_data.truncated,
                     lambda_ = gae_lambda,
                     gamma = discounting_factor)
    assert advantages.shape == values_excl_last.shape
    advantages = jax.lax.stop_gradient(advantages)
    target_values = jax.lax.stop_gradient(values_excl_last + advantages)

    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = jax.lax.stop_gradient(advantages)

    old_loglikelihoods = jax.lax.stop_gradient(rollout_data.network_output.loglikelihoods)
    assert network_output.loglikelihoods.shape == advantages.shape
    assert old_loglikelihoods.shape == advantages.shape
    likelihood_ratios = jp.exp(network_output.loglikelihoods - old_loglikelihoods)
    loss_cand1 = likelihood_ratios * advantages
    loss_cand2 = jp.clip(likelihood_ratios, 1 - clip_range, 1 + clip_range) * advantages

    # Note that it's the network's responsiblity to add entropy loss as one particular
    # instance of a regularization loss.
    assert network_output.value_estimates.shape == target_values.shape
    actor_loss = -jp.mean(jp.minimum(loss_cand1, loss_cand2))
    critic_loss = 0.5 * jp.mean((network_output.value_estimates - target_values)**2)
    regularization_loss = jp.mean(network_output.regularization_loss)
    
    if LoggingLevel.LOSSES in logging_level:
        metrics["losses/actor"] = actor_loss
        metrics["losses/critic"] = critic_loss
        metrics["losses/regularization"] = regularization_loss
    if LoggingLevel.ACTOR_EXTRA in logging_level:
            metrics["correlations/advantage_action"] = jp.corrcoef(advantages.flatten(),
                                                                   rollout_data.network_output.actions.flatten())[0, 1]
            metrics["correlations/ll_advantage"] = jp.corrcoef(rollout_data.network_output.loglikelihoods.flatten(),
                                                    advantages.flatten())[0, 1]
            _log_metric(metrics, "likelihood ratios", likelihood_ratios, logging_percentiles)
            metrics["clipping_fraction"] = jp.mean(jp.logical_or(likelihood_ratios<1-clip_range, likelihood_ratios>1+clip_range))
    if LoggingLevel.CRITIC_EXTRA in logging_level:
        _log_metric(metrics, "losses/predicted_value", rollout_data.network_output.value_estimates, logging_percentiles)
        _log_metric(metrics, "advantages", advantages, logging_percentiles)
        metrics["losses/critic_R^2"] = 1.0 - 2 * metrics["losses/critic"] / (jp.var(target_values) + 1e-8)

    total_loss = actor_loss + critic_loss + regularization_loss

    return total_loss, metrics

def new_training_state(env: mujoco_playground.MjxEnv,
                       networks: PPONetwork,
                       n_envs: int,
                       seed: int,
                       learning_rate: float=1e-4):
    # Setup keys
    key = jax.random.key(seed)
    key, training_key = jax.random.split(key)
 
    # Setup environment states
    env_init_keys = jax.random.split(key, n_envs)
    env_states = nnx.vmap(env.reset)(env_init_keys)

    # Setup network states
    network_states = networks.initialize_state(n_envs)

    # Setup optimizer
    optimizer = nnx.Optimizer(networks, optax.adam(learning_rate=learning_rate), wrt=nnx.Param)
    return TrainingState(networks, network_states, env_states,
                         optimizer, training_key, jp.array(0.0))

def checkify_tree_equals(A, B, msg: str):
    jax.tree.map(lambda a,b: checkify.check(jp.all(a == b), msg), A, B)
