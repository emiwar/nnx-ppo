from typing import Union, Dict, Tuple, Any
import functools

from flax import struct, nnx
import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env

import networks
import networks.types

@struct.dataclass
class Transition:
  """Environment state for training and inference."""
  obs: mjx_env.Observation
  network_output: networks.types.PPONetworkOutput
  rewards: Union[Dict, jax.Array]
  done: Union[Dict, jax.Array]
  next_obs:mjx_env.Observation

def single_transition(env: mjx_env.MjxEnv,
                      networks: networks.types.AbstractPPOActorCritic,
                      carry: Tuple[Dict, mjx_env.State],
                      rng_key_for_env_reset: jax.Array):
  network_state, env_state = carry
  next_network_state, network_output = networks(network_state, env_state.obs)
  next_env_state = env.step(env_state, network_output.actions)
  transition = Transition(obs=env_state.obs,
                          network_output=network_output,
                          rewards=next_env_state.reward,
                          done=next_env_state.done,
                          next_obs=next_env_state.obs)
  next_network_state = jax.lax.cond(transition.done, networks.reset_state,
                                    lambda s: s, next_network_state)
  next_env_state = jax.lax.cond(transition.done, env.reset,
                                lambda rng: next_env_state, rng_key_for_env_reset)

  return (next_network_state, next_env_state), transition

def unroll_env(env: mjx_env.MjxEnv,
               env_state: mjx_env.State,
               networks: networks.types.AbstractPPOActorCritic,
               network_state: Dict,
               unroll_length: int,
               rng_key_for_env_reset: jax.Array):
  rng_keys_for_env_reset = jax.random.split(rng_key_for_env_reset, unroll_length)
  step = functools.partial(single_transition, env)
  (final_network_state, final_env_state), unroll_seq = nnx.scan(step,
    in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0),
    out_axes=(nnx.Carry, 0),
    length=unroll_length)(
      networks,
      (network_state, env_state),
      rng_keys_for_env_reset
    )
  return final_network_state, final_env_state, unroll_seq
