from typing import Any, Optional, Tuple, Dict, Union, Mapping

from ml_collections import config_dict
import mujoco_playground

from flax import nnx
import jax
import jax.numpy as jp
from jax.experimental import checkify
import optax

from nnx_ppo.networks.types import PPONetwork
from nnx_ppo.algorithms import rollout
from nnx_ppo.algorithms.types import TrainingState, LoggingLevel

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
        gradient_clipping = None,
        weight_decay = None,
    )

def train_ppo(env: mujoco_playground.MjxEnv,
              networks: PPONetwork,
              config: config_dict.ConfigDict = default_config(),
              seed: int = 17,
              logging_level: LoggingLevel = LoggingLevel.BASIC):
    training_state = new_training_state(env, networks, config.n_envs, seed,
                                        config.learning_rate,
                                        config.gradient_clipping,
                                        config.weight_decay)
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

    loss_metrics = scan_update(training_state.networks, training_state.optimizer, all_indices)
    total_steps = training_state.steps_taken + rollout_length * n_envs
    metrics = compute_metrics(loss_metrics, rollout_data, logging_level, logging_percentiles)
    metrics["total_steps"] = total_steps
    if LoggingLevel.WEIGHTS in logging_level:
        _log_weight_stats(metrics, training_state.networks, logging_percentiles)
    training_state.networks.update_statistics(rollout_data, total_steps)

    # Now that all updates are done, we can replace all the network (and environment)
    # states in training state. Note that this would have been incorrect to update
    # earlier (see note above).
    training_state = training_state.replace(
        network_states = next_net_state,
        env_states = next_env_state,
        rng_key = new_key,
        steps_taken = total_steps,
    )
    
    return training_state, metrics

def compute_metrics(loss_metrics: Dict[str, jax.Array],
                    rollout_data: rollout.Transition,
                    logging_level: LoggingLevel,
                    percentile_levels: Optional[Tuple] = None):
    metrics = {}
    for k, v in loss_metrics.items():
        _log_metric(metrics, k, v, percentile_levels)
    if LoggingLevel.TRAINING_ENV_METRICS in logging_level:
        for k, v in rollout_data.metrics.items():
            _log_metric(metrics, k, v, percentile_levels)
    if LoggingLevel.TRAIN_ROLLOUT_STATS in logging_level:
        _log_metric(metrics, "rollout_batch/reward", rollout_data.rewards, percentile_levels)
        _log_metric(metrics, "rollout_batch/action", rollout_data.network_output.actions, percentile_levels)
        metrics["rollout_batch/done_rate"] = rollout_data.done.mean()
        metrics["rollout_batch/truncation_rate"] = rollout_data.truncated.mean()
        metrics["rollout_batch/obs_NaN"] = 1.0 - jp.isfinite(rollout_data.obs).mean()
        metrics["rollout_batch/next_obs_NaN"] = 1.0 - jp.isfinite(rollout_data.next_obs).mean()
    if LoggingLevel.ACTOR_EXTRA in logging_level:
        _log_metric(metrics, "loglikelihood", rollout_data.network_output.loglikelihoods, percentile_levels)
        if rollout_data.network_output.actions.shape[-1] == 1:
            metrics["correlations/action_ll"] = jp.corrcoef(rollout_data.network_output.loglikelihoods.flatten(),
                                                    rollout_data.network_output.actions.flatten())[0, 1]
    if LoggingLevel.CRITIC_EXTRA in logging_level:
        _log_metric(metrics, "losses/predicted_value", rollout_data.network_output.value_estimates, percentile_levels)
    return metrics

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


def _log_weight_stats(metrics: Dict[str, jax.Array],
                      networks: PPONetwork,
                      percentile_levels: Optional[Tuple] = None):
    """Log weight statistics for actor and critic networks separately."""
    # Extract parameters using nnx.state
    actor_params = nnx.state(networks.actor, nnx.Param)
    critic_params = nnx.state(networks.critic, nnx.Param)

    # Flatten all actor weights into single array
    actor_weights = jp.concatenate([p.flatten() for p in jax.tree.leaves(actor_params)])
    critic_weights = jp.concatenate([p.flatten() for p in jax.tree.leaves(critic_params)])

    _log_metric(metrics, "weights/actor", actor_weights, percentile_levels)
    _log_metric(metrics, "weights/critic", critic_weights, percentile_levels)


def gae(rewards, values, done, truncation, lambda_: float, gamma: float):
    assert values.shape == (rewards.shape[0]+1, rewards.shape[1])
    assert rewards.shape == done.shape
    assert truncation.shape == done.shape
    def inner_step(next_advantage, reward, old_value, next_value, done, truncated):
        next_value = jp.where(done, 0.0, next_value)
        new_value = reward + gamma * next_value
        advantage = new_value - old_value
        advantage = jp.where(truncated, 0.0, advantage)
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
    @jax.vmap
    def reset_net_state(done, state):
        return jax.lax.cond(done, networks.initialize_state, lambda _: state, done.shape)
    
    def step_network(networks: PPONetwork, net_state, obs, done, raw_action):
        net_state, network_output = networks(net_state, obs, raw_action)
        net_state = reset_net_state(done, net_state)
        return net_state, network_output

    time_scan = nnx.scan(step_network, in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0, 0, 0), out_axes=(nnx.Carry, 0))
    next_net_state_again, network_output = time_scan(networks, network_state, rollout_data.obs, rollout_data.done, rollout_data.network_output.raw_actions)
    
    if network_output.value_estimates.ndim == 3:
        assert network_output.value_estimates.shape[2] == 1
        network_output = network_output.replace(
            value_estimates = network_output.value_estimates[:, :, 0]
        )
    last_obs = jax.tree.map(lambda x: x[-1], rollout_data.next_obs)
    _, network_output_last = networks(next_net_state_again, last_obs)
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
    
    loss_metrics = dict()
    if LoggingLevel.LOSSES in logging_level:
        loss_metrics["losses/actor"] = actor_loss
        loss_metrics["losses/critic"] = critic_loss
        loss_metrics["losses/regularization"] = regularization_loss
    if LoggingLevel.ACTOR_EXTRA in logging_level:
        loss_metrics["correlations/ll_advantage"] = jp.corrcoef(rollout_data.network_output.loglikelihoods.flatten(), advantages.flatten())[0, 1]
        loss_metrics["losses/likelihood_ratios"] = likelihood_ratios
        loss_metrics["losses/likelihood_ratios_mean"] = jp.mean(likelihood_ratios)
        loss_metrics["losses/clipping_fraction"] = jp.mean(jp.logical_or(likelihood_ratios<1-clip_range, likelihood_ratios>1+clip_range))
        loss_metrics["losses/new_loglikelihoods"] = network_output.loglikelihoods
        loss_metrics["losses/loglikelihood_diff"] = network_output.loglikelihoods - old_loglikelihoods
        loss_metrics["losses/new_mu"] = network_output.metrics["action_sampler"]["mu"]
        loss_metrics["losses/new_sigma"] = network_output.metrics["action_sampler"]["sigma"]
        loss_metrics["losses/mu_diff"] = network_output.metrics["action_sampler"]["mu"] - rollout_data.network_output.metrics["action_sampler"]["mu"]
        loss_metrics["losses/sigma_diff"] = network_output.metrics["action_sampler"]["sigma"] - rollout_data.network_output.metrics["action_sampler"]["sigma"]
        loss_metrics["losses/sigma_ratio"] = network_output.metrics["action_sampler"]["sigma"] / rollout_data.network_output.metrics["action_sampler"]["sigma"]
    if LoggingLevel.CRITIC_EXTRA in logging_level:
        loss_metrics["losses/predicted_value"] = values_excl_last
        loss_metrics["losses/advantages"] = advantages
        loss_metrics["losses/advantages_NaN"] = 1.0 - jp.isfinite(advantages).mean()
        loss_metrics["losses/critic_R^2"] = 1.0 - 2 * critic_loss / (jp.var(target_values) + 1e-8)

    total_loss = actor_loss + critic_loss + regularization_loss

    #Sometimes, for some inexplicable reason, the network produces garbage outputs
    #during this function, but not during earlier rollouts. So a heuristic is that
    #if the _mean_ likelihood ratio is out of clipping bounds, the minibatch is bad
    #and we just ignore it by setting the loss to 0.0. 
    #total_loss *= jp.median(likelihood_ratios) > (1-clip_range)
    #total_loss *= jp.median(likelihood_ratios) < (1+clip_range)

    return total_loss, loss_metrics

def new_training_state(env: mujoco_playground.MjxEnv,
                       networks: PPONetwork,
                       n_envs: int,
                       seed: int,
                       learning_rate: float=1e-4,
                       gradient_clipping: Optional[float] = None,
                       weight_decay: Optional[float] = None):
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
        #Optax default decay
        optimizer_chain_links.append(optax.adamw(learning_rate=learning_rate))
    else:
        optimizer_chain_links.append(optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay))
    optimizer = nnx.Optimizer(networks, optax.chain(*optimizer_chain_links), wrt=nnx.Param)
    return TrainingState(networks, network_states, env_states,
                         optimizer, training_key, jp.array(0.0))

def checkify_tree_equals(A, B, msg: str):
    jax.tree.map(lambda a,b: checkify.check(jp.all(a == b), msg), A, B)
