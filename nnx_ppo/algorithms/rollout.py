from typing import Union, Dict, Tuple, Any
import functools

from flax import struct, nnx
import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env

from nnx_ppo import networks
import nnx_ppo.networks.types

@struct.dataclass
class Transition:
  """Environment state for training and inference."""
  obs: mjx_env.Observation
  network_output: networks.types.PPONetworkOutput
  rewards: Union[Dict, jax.Array]
  done: Union[Dict, jax.Array]
  truncated: Union[Dict, jax.Array]
  next_obs: mjx_env.Observation

def single_transition(env: mjx_env.MjxEnv,
                      networks: nnx_ppo.networks.types.AbstractPPOActorCritic,
                      carry: Tuple[Dict, mjx_env.State],
                      rng_key_for_env_reset: jax.Array):
  network_state, env_state = carry
  next_network_state, network_output = networks(network_state, env_state.obs)
  next_env_state = env.step(env_state, network_output.actions)
  transition = Transition(obs=env_state.obs,
                          network_output=network_output,
                          rewards=next_env_state.reward,
                          done=next_env_state.done,
                          truncated=next_env_state.info.get("truncated", False),
                          next_obs=next_env_state.obs)
  next_network_state = jax.lax.cond(transition.done, networks.reset_state,
                                    lambda s: s, next_network_state)
  next_env_state = jax.lax.cond(transition.done, env.reset,
                                lambda rng: next_env_state, rng_key_for_env_reset)

  return (next_network_state, next_env_state), transition

def unroll_env(env: mjx_env.MjxEnv,
               env_state: mjx_env.State,
               networks: nnx_ppo.networks.types.AbstractPPOActorCritic,
               network_state: Dict,
               unroll_length: int,
               rng_key_for_env_reset: jax.Array):
  rng_keys_for_env_reset = jax.random.split(rng_key_for_env_reset, unroll_length)
  step = functools.partial(single_transition, env)
  (final_network_state, final_env_state), rollout = nnx.scan(step,
    in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0),
    out_axes=(nnx.Carry, 0),
    length=unroll_length)(
      networks,
      (network_state, env_state),
      rng_keys_for_env_reset
    )
  if rollout.network_output.value_estimates.ndim == 2:
    assert rollout.network_output.value_estimates.shape[1] == 1
    rollout = rollout.replace(
      network_output = rollout.network_output.replace(
        value_estimates = rollout.network_output.value_estimates[:, 0]
      )
    )
  assert rollout.network_output.value_estimates.shape == rollout.rewards.shape
  return final_network_state, final_env_state, rollout

def eval_rollout(env: mjx_env.MjxEnv,
                 networks: nnx_ppo.networks.types.AbstractPPOActorCritic,
                 n_envs: int,
                 max_episode_length: int,
                 key: jax.Array):
  net_key, env_key = jax.random.split(key)
  env_keys = jax.random.split(env_key, n_envs)
  env_states = jax.vmap(env.reset)(env_keys)
  net_init_keys = jax.random.split(net_key, n_envs)
  net_states = nnx.vmap(networks.initialize_state)(net_init_keys)

  def step(env, networks, carry):
    env_state, network_state, cuml_reward, lifespan = carry
    next_network_state, network_output = networks(network_state, env_state.obs)
    next_env_state = env.step(env_state, network_output.actions)
    cuml_reward += jp.where(next_env_state.done, 0.0, next_env_state.reward)
    lifespan += jp.logical_not(next_env_state.done).astype(float)
    return next_env_state, next_network_state, cuml_reward, lifespan
  step_partial = functools.partial(step, env)
  step_scan = nnx.scan(step_partial,
                       in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
                       out_axes=nnx.Carry,
                       length = max_episode_length)
  step_vmap = nnx.vmap(step_scan, in_axes=(None, 0), out_axes=0)
  init_carry = (env_states, net_states, env_states.reward, jp.zeros(n_envs))
  _, _, cuml_reward, lifespan = step_vmap(networks, init_carry)
  return dict(
    episode_reward_mean = cuml_reward.mean(),
    episode_reward_std = cuml_reward.std(),
    lifespan_mean = lifespan.mean(),
    lifespan_std = lifespan.std(),
  )