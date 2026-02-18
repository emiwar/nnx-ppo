import mujoco_playground
import functools
from flax import nnx
import jax

import networks.modules
import algorithms.rollout as rollout


SEED = 17
N_ENVS = 8

env = mujoco_playground.registry.load("CartpoleBalance")
nets = networks.modules.MLPActorCritic(env.observation_size, env.action_size,
                                   actor_hidden_sizes=[16, 16],
                                   critic_hidden_sizes=[16, 16],
                                   rngs = nnx.Rngs(SEED, action_sampling=SEED))
key = jax.random.key(SEED+1)
reset_keys = jax.random.split(key, N_ENVS)
states = jax.vmap(env.reset)(reset_keys)


#import importlib; importlib.reload(rollout)
#unroll = lambda states, nets, L: rollout.unroll_env(env, states, nets, L)
unroll_vmap = nnx.jit(nnx.vmap(rollout.unroll_env, in_axes=(None, 0, None, None)), static_argnums=(0, 3))
#unroll_vmap = nnx.vmap(rollout.unroll_env, in_axes=(None, 0, None, None))
final_state, data = unroll_vmap(env, states, nets, 4)


single_state = env.reset(key)
#final_state, transition = rollout.single_transition(env, single_state, nets)

step_vmap = nnx.jit(nnx.vmap(rollout.single_transition, in_axes=(None, 0, None)), static_argnums=(0))
step_vmap(env, states, nets)

rollout.unroll_env(env, single_state, nets, 7)
