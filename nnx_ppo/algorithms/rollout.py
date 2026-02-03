from typing import Union, Dict, Tuple, Any
import functools

from flax import struct, nnx
import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env
import nnx_ppo.networks.types

@struct.dataclass
class Transition:
  """Environment state for training and inference."""
  obs: mjx_env.Observation
  network_output: nnx_ppo.networks.types.PPONetworkOutput
  rewards: Union[Dict, jax.Array]
  done: jax.Array
  truncated: jax.Array
  next_obs: mjx_env.Observation
  metrics: Dict[str, Any]

def single_transition(env: mjx_env.MjxEnv,
                      networks: nnx_ppo.networks.types.PPONetwork,
                      carry: Tuple[Dict, mjx_env.State],
                      rng_keys_for_env_reset: jax.Array):
  network_state, env_state = carry
  next_network_state, network_output = networks(network_state, env_state.obs)
  next_env_state = jax.vmap(env.step)(env_state, network_output.actions)
  transition = Transition(obs=env_state.obs,
                          network_output=network_output,
                          rewards=next_env_state.reward,
                          done=next_env_state.done,
                          truncated=next_env_state.info.get("truncated", jp.zeros(next_env_state.done.shape, jp.bool)),
                          next_obs=next_env_state.obs,
                          metrics={
                            "env": next_env_state.metrics,
                            "net": network_output.metrics,
                          })
  @jax.vmap
  def reset_env_state(done, state, rng):
    return jax.lax.cond(done, env.reset, lambda rng: state, rng)
  next_env_state = reset_env_state(transition.done,
                                   next_env_state,
                                   rng_keys_for_env_reset)
  
  @jax.vmap
  def reset_net_state(done, state):
    return jax.lax.cond(done, networks.initialize_state, lambda _: state, done.shape)
  next_network_state = reset_net_state(transition.done,
                                       next_network_state)
  return (next_network_state, next_env_state), transition

def unroll_env(env: mjx_env.MjxEnv,
               env_state: mjx_env.State,
               networks: nnx_ppo.networks.types.PPONetwork,
               network_state: Any,
               unroll_length: int,
               rng_key_for_env_reset: jax.Array):
  batch_size = env_state.done.shape[0]
  rng_keys_for_env_reset = jax.random.split(rng_key_for_env_reset, (unroll_length, batch_size))
  step = functools.partial(single_transition, env)
  (final_network_state, final_env_state), rollout = nnx.scan(step,
    in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0),
    out_axes=(nnx.Carry, 0),
    length=unroll_length)(
      networks,
      (network_state, env_state),
      rng_keys_for_env_reset
    )
  if rollout.network_output.value_estimates.ndim == 3:
    assert rollout.network_output.value_estimates.shape[-1] == 1
    rollout = rollout.replace(
      network_output = rollout.network_output.replace(
        value_estimates = rollout.network_output.value_estimates[..., 0]
      )
    )
  assert rollout.network_output.value_estimates.shape == rollout.rewards.shape
  return final_network_state, final_env_state, rollout

def eval_rollout(env: mjx_env.MjxEnv,
                 networks: nnx_ppo.networks.types.PPONetwork,
                 n_envs: int,
                 max_episode_length: int,
                 key: jax.Array):
  env_keys = jax.random.split(key, n_envs)
  env_states = jax.vmap(env.reset)(env_keys)
  net_states = networks.initialize_state(n_envs)

  def step(env, networks, carry):
    env_state, network_state, cuml_reward, lifespan = carry
    next_network_state, network_output = networks(network_state, env_state.obs, gradient_mode=True)
    next_env_state = jax.vmap(env.step)(env_state, network_output.actions)
    next_env_state = next_env_state.replace(done = jp.logical_or(next_env_state.done, env_state.done).astype(float))
    # Only accumulate reward if env was not already done before this step
    cuml_reward += jp.where(env_state.done, 0.0, next_env_state.reward)
    lifespan += jp.logical_not(next_env_state.done).astype(float)
    return next_env_state, next_network_state, cuml_reward, lifespan
  step_partial = functools.partial(step, env)
  step_scan = nnx.scan(step_partial,
                       in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
                       out_axes=nnx.Carry,
                       length = max_episode_length)
  init_carry = (env_states, net_states, env_states.reward, jp.zeros(n_envs))
  _, _, cuml_reward, lifespan = step_scan(networks, init_carry)
  return dict(
    episode_reward_mean = cuml_reward.mean(),
    episode_reward_std = cuml_reward.std(),
    lifespan_mean = lifespan.mean(),
    lifespan_std = lifespan.std(),
  )

def eval_rollout_for_render_scan(env: mjx_env.MjxEnv,
                                  networks: nnx_ppo.networks.types.PPONetwork,
                                  max_episode_length: int,
                                  key: jax.Array):
  """JIT-compatible scan-based rollout that returns stacked states.

  Returns:
    stacked_states: State pytree with leading dimension of max_episode_length.
    final_state: The final environment state.
    total_reward: Total reward accumulated during the episode.
  """
  key, key2 = jax.random.split(key)
  env_state = env.reset(key)
  net_state = networks.initialize_state(1)
  net_state = jax.tree.map(lambda x: x[0], net_state)

  def step_fn(networks, carry):
    env_state, net_state, cumulative_reward, already_done, rng = carry

    obs_batched = jax.tree.map(lambda x: x[None], env_state.obs)
    net_state_batched = jax.tree.map(lambda x: x[None], net_state)
    next_net_state, network_output = networks(net_state_batched, obs_batched, gradient_mode=True)
    next_net_state = jax.tree.map(lambda x: x[0], next_net_state)
    action = network_output.actions[0]

    next_env_state = env.step(env_state, action)
    # Only accumulate reward if not already done
    new_cumulative_reward = cumulative_reward + jp.where(already_done, 0.0, next_env_state.reward)
    new_already_done = jp.logical_or(already_done, next_env_state.done)
    next_env_state = jax.lax.cond(next_env_state.done, env.reset, lambda rng: next_env_state, rng)
    next_net_state = jax.lax.cond(next_env_state.done, networks.initialize_state, lambda _: next_net_state, next_env_state.done.shape)

    new_rng, = jax.random.split(rng, 1)
    return (next_env_state, next_net_state, new_cumulative_reward, new_already_done, new_rng), env_state

  scan_fn = nnx.scan(
    step_fn,
    in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
    out_axes=(nnx.Carry, 0),
    length=max_episode_length
  )

  init_carry = (env_state, net_state, jp.array(0.0), jp.array(False), key2)
  (final_env_state, _, total_reward, _, _), stacked_states = scan_fn(networks, init_carry)

  return stacked_states, final_env_state, total_reward


def unstack_trajectory(stacked_states, final_state, max_episode_length: int):
  """Convert stacked states from scan to a list for rendering.

  This must be called outside of JIT since it creates a Python list.
  """
  trajectory = [jax.tree.map(lambda x: x[i], stacked_states) for i in range(max_episode_length)]
  trajectory.append(final_state)
  return trajectory