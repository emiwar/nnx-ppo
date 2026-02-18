import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from datetime import datetime

import jax
import jax.numpy as jp
import numpy as np
from flax import nnx
import wandb

from vnl_playground.tasks.rodent.imitation import Imitation
from vnl_playground.tasks.wrappers import FlattenObsWrapper

from nnx_ppo.networks.feedforward import MLPActorCritic
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.algorithms import ppo, rollout

jax.config.update("jax_debug_nans", True)

SEED = 40
train_env = FlattenObsWrapper(Imitation())
eval_env = train_env

rngs = nnx.Rngs(SEED)
nets = MLPActorCritic(train_env.observation_size, train_env.action_size,
                      actor_hidden_sizes=[256,] * 4,
                      critic_hidden_sizes=[512,] * 2,
                      rngs=rngs,
                      transfer_function=nnx.swish,
                      action_sampler=NormalTanhSampler(rngs, entropy_weight=2e-3, min_std=5e-3, std_scale=1.0, preclamp=False),
                      normalize_obs=True)
reset_key = jax.random.key(42)
env_state = jax.jit(train_env.reset)(reset_key)
net_state = jax.jit(nets.initialize_state)(1)
step_env = jax.jit(train_env.step)

for i in range(200):
    print(f"Iter {i}")
    cache_size = step_env._cache_size()
    print(f"\t step_env cache size: {cache_size}")
    net_state, net_output = nets(net_state, env_state.obs)
    if jp.any(jp.isnan(net_output.actions)):
        raise ValueError("Net produces NaN actions")
    actions = net_output.actions
    #actions = jax.random.uniform(jax.random.key(42+i+1), (38,))#net_output.actions
    next_env_state = step_env(env_state, actions)

    def_1 = jax.tree.structure(env_state)
    def_2 = jax.tree.structure(next_env_state)
    if def_1 != def_2:
        raise AssertionError(f"env_state changed from {def_1} to {def_2}")

    if jp.any(jp.isnan(next_env_state.obs)):
        raise ValueError("Env produces NaN obs")
    any_metrics_nan = False
    for k,m in next_env_state.metrics.items():
        if jp.any(jp.isnan(m)):
            print(f"metrics['{k}'] contains NaN")
            any_metrics_nan = True
    if any_metrics_nan:
        raise ValueError("Env has NaN in metrics")
    if next_env_state.done:
        print("\tEnv is done.")
        for k,v in next_env_state.metrics.items():
            if k.startswith("termination"):
                print(f"\t{k}: {v}")
    env_state = next_env_state