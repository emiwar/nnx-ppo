from typing import Any, Optional, Tuple, Dict
import enum

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
        normalize_advantages = True,
    )

@flax.struct.dataclass
class TrainingState:
    networks: PPONetwork
    network_states: Any
    env_states: mujoco_playground.State
    optimizer: nnx.Optimizer
    rng_key: jax.Array
    steps_taken: int

class LoggingLevel(enum.Flag):
    LOSSES = enum.auto()
    CRITIC_EXTRA = enum.auto()
    ACTOR_EXTRA = enum.auto()
    TRAIN_ROLLOUT_STATS = enum.auto()
    ASSERTS = enum.auto()
    BASIC = LOSSES
    ALL = LOSSES | ACTOR_EXTRA | CRITIC_EXTRA | TRAIN_ROLLOUT_STATS | ASSERTS
    NONE = 0

def train_ppo(env: mujoco_playground.MjxEnv,
              networks: PPONetwork,
              config: config_dict.ConfigDict = default_config(),
              seed: int = 17,
              logging_level: LoggingLevel = LoggingLevel.BASIC):
    training_state = new_training_state(env, networks, config.n_envs, seed)
    ppo_step_jit = nnx.jit(ppo_step, static_argnums=(0, 2, 3, 7, 8))
    while training_state.steps_taken < config.n_steps:
        training_state, metrics = ppo_step_jit(
            env, training_state, 
            config.n_envs, config.rollout_length,
            config.gae_lambda, config.discounting_factor,
            config.clip_range, config.normalize_advantages,
            logging_level
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
             logging_level: LoggingLevel = LoggingLevel.LOSSES) -> Tuple[TrainingState, Dict]:
    reset_key, new_key = jax.random.split(training_state.rng_key)
    pre_rollout_module_state = nnx.state(training_state.networks)
    next_net_state, next_env_state, rollout_data = rollout.unroll_env(
        env,
        training_state.env_states,
        training_state.networks,
        training_state.network_states,
        rollout_length,
        reset_key
    )
    if LoggingLevel.ASSERTS in logging_level:
        post_rollout_module_state = nnx.state(training_state.networks)

    # We need the value of the final observation
    last_obs = jax.tree.map(lambda x: x[-1], rollout_data.next_obs)
    _, network_output = training_state.networks(next_net_state, last_obs)
    last_value = network_output.value_estimates
    last_value = last_value.reshape((1, last_value.shape[0]))
    values_excl_last = rollout_data.network_output.value_estimates
    values_incl_last = jp.concatenate((values_excl_last, last_value), axis=0)

    advantages = gae(rewards=rollout_data.rewards,
                     values=values_incl_last,
                     done=rollout_data.done,
                     truncation=rollout_data.truncated,
                     lambda_ = gae_lambda,
                     gamma = discounting_factor)
    assert advantages.shape == values_excl_last.shape
    target_values = values_excl_last + advantages

    # Rollback any module state changes that happened during the rollout, including
    # RNG state. Second, we use the network states from training_state, which has not
    # yet been updated to next_net_state. This ensures the updates start from the same
    # `network_states` as the rollouts did, which is necessary for the gradients
    # to be correct.
    nnx.update(training_state.networks, pre_rollout_module_state)
    grads, metrics = nnx.grad(ppo_loss, has_aux=True)(
        networks = training_state.networks,
        network_state = training_state.network_states,
        observations = rollout_data.obs,
        actions = rollout_data.network_output.actions,
        old_loglikelihoods = rollout_data.network_output.loglikelihoods,
        target_values = target_values,
        advantages = advantages,
        next_net_state = next_net_state,
        clip_range = clip_range,
        normalize_advantages = normalize_advantages,
        logging_level = logging_level
    )
    if LoggingLevel.TRAIN_ROLLOUT_STATS in logging_level:
        metrics["reward_mean"] = rollout_data.rewards.mean()
        metrics["reward_std"] = rollout_data.rewards.std()
        metrics["advantage/mean"] = advantages.mean()
        metrics["advantage/std"] = advantages.std()
        metrics["advantage/min"] = advantages.min()
        metrics["advantage/max"] = advantages.max()
        metrics["action_mean"] = rollout_data.network_output.actions.mean()
        metrics["action_std"] = rollout_data.network_output.actions.std()
    if LoggingLevel.ASSERTS in logging_level:
        post_update_module_state = nnx.state(training_state.networks)
        metrics["asserts/module_state_identical"] = jax.tree.reduce(jp.logical_and, jax.tree.map(jp.allclose, post_rollout_module_state, post_update_module_state), jp.array(True)).astype(int)
    training_state.optimizer.update(grads)

    # Now that all updates are done, we can replace all the network (and environment)
    # states in training state. Note that this would have been incorrect to update
    # earlier (see note above).
    training_state = training_state.replace(
        network_states = next_net_state,
        env_states = next_env_state,
        rng_key = new_key,
        steps_taken = training_state.steps_taken + rollout_length * n_envs,
    )
    return training_state, metrics

def gae(rewards, values, done, truncation, lambda_: float, gamma: float):
    assert values.shape == (rewards.shape[0]+1, rewards.shape[1])
    assert rewards.shape == done.shape
    def inner_step(next_advantage, reward, old_value, next_old_value, done, truncated):
        next_value = jp.where(done,
            jp.where(truncated, old_value, jp.zeros_like(next_old_value)),
            next_old_value
        )
        new_value = reward + gamma * next_value
        advantage = new_value - old_value
        gae_advantage = advantage + jp.logical_not(done) * gamma * lambda_ * next_advantage
        return gae_advantage, gae_advantage
    time_scan = nnx.scan(inner_step,
        in_axes=(nnx.Carry, 0, 0, 0, 0, 0), out_axes=(nnx.Carry, 0),
        length=rewards.shape[0], reverse=True)
    _, advantages = time_scan(next_advantage = jp.zeros(rewards.shape[1]),
                              reward=rewards,
                              old_value=values[:-1, :],
                              next_old_value=values[1:, :],
                              done=done,
                              truncated=truncation)
    return advantages

def ppo_loss(networks: PPONetwork, network_state,
             observations, actions, old_loglikelihoods, target_values,
             advantages, next_net_state, clip_range, normalize_advantages,
             logging_level: LoggingLevel):
    metrics = dict()

    time_scan = nnx.scan(lambda networks, net_state, obs: networks(net_state, obs),
                         in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0),
                         out_axes=(nnx.Carry, 0))
    next_net_state_again, network_output = time_scan(networks, network_state, observations)
    
    if network_output.value_estimates.ndim == 3:
        assert network_output.value_estimates.shape[2] == 1
        network_output = network_output.replace(
            value_estimates = network_output.value_estimates[:, :, 0]
        )

    if LoggingLevel.ASSERTS in logging_level:
        metrics["asserts/actions_max_diff"] = jp.max(jp.abs(network_output.actions - actions))
        metrics["asserts/likelihoods_max_diff"] = jp.max(jp.abs(network_output.loglikelihoods - old_loglikelihoods))
        metrics["asserts/critic_values_max_diff"] = jp.max(jp.abs(network_output.value_estimates - (target_values-advantages)))
        metrics["asserts/net_state_identical"] = jax.tree.reduce(jp.logical_and, jax.tree.map(jp.allclose, next_net_state_again, next_net_state), jp.array(True)).astype(int)

    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    assert network_output.loglikelihoods.shape == advantages.shape
    assert old_loglikelihoods.shape == advantages.shape
    likelihood_ratios = jp.exp(network_output.loglikelihoods - old_loglikelihoods)
    loss_cand1 = likelihood_ratios * advantages
    loss_cand2 = jp.clip(likelihood_ratios, 1 - clip_range, 1 + clip_range) * advantages

    # Note that it's the network's responsiblity to add entropy loss as one particular
    # instance of a regularization loss.
    actor_loss = -jp.mean(jp.minimum(loss_cand1, loss_cand2))
    critic_loss = 0.5 * jp.mean((network_output.value_estimates - target_values)**2)
    regularization_loss = jp.mean(network_output.regularization_loss)

    if LoggingLevel.LOSSES in logging_level:
        metrics["losses/actor"] = actor_loss
        metrics["losses/critic"] = critic_loss
        metrics["losses/regularization"] = regularization_loss
    if LoggingLevel.ACTOR_EXTRA in logging_level:
        metrics["loglikelihood/mean"] = jp.mean(network_output.loglikelihoods)
        metrics["loglikelihood/std"] = jp.std(network_output.loglikelihoods)
        metrics["loglikelihood/min"] = jp.min(network_output.loglikelihoods)
        metrics["loglikelihood/max"] = jp.max(network_output.loglikelihoods)
    if LoggingLevel.CRITIC_EXTRA in logging_level:
        metrics["losses/predicted_value_mean"] = jp.mean(network_output.value_estimates)
        metrics["losses/predicted_value_std"] = jp.std(network_output.value_estimates)
        metrics["losses/target_value_mean"] = jp.mean(target_values)
        metrics["losses/target_value_std"] = jp.std(target_values)
        metrics["losses/critic_R^2"] = 1.0 - 2 * critic_loss / jp.var(target_values)

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
    env_init_key, net_init_key = jax.random.split(key)
 
    # Setup environment states
    env_init_keys = jax.random.split(env_init_key, n_envs)
    env_states = nnx.vmap(env.reset)(env_init_keys)

    # Setup network states
    net_init_keys = jax.random.split(net_init_key, n_envs)
    network_states = nnx.vmap(networks.initialize_state)(net_init_keys)

    # Setup optimizer
    optimizer = nnx.Optimizer(networks, optax.adam(learning_rate=learning_rate), wrt=nnx.Param)
    return TrainingState(networks, network_states, env_states,
                         optimizer, training_key, 0)

def checkify_tree_equals(A, B, msg: str):
    jax.tree.map(lambda a,b: checkify.check(jp.all(a == b), msg), A, B)