from datetime import datetime
import functools

import pandas as pd
import numpy as np
from ml_collections import config_dict
import jax

import mujoco_playground
import mujoco_playground.config.dm_control_suite_params

import brax
import brax.training.agents.ppo.train
import brax.training.agents.ppo.networks

from flax import nnx
from nnx_ppo.networks.modules import MLPActorCritic
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.algorithms import ppo, rollout
from nnx_ppo.wrappers import reward_scaling_wrapper

env_name = 'CartpoleSwingup'

env = mujoco_playground.registry.load(env_name)
env_cfg = mujoco_playground.registry.get_default_config(env_name)
ppo_params = mujoco_playground.config.dm_control_suite_params.brax_ppo_config(env_name)
ppo_params.num_evals = 100

'''
print("BRAX")
x_data, y_data, y_dataerr = [np.nan], [np.nan], [np.nan]
times = [datetime.now()]
def progress(num_steps, metrics):
  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])
  print(f"{times[-1]} ({num_steps}): {y_data[-1]}")

network_factory = brax.training.agents.ppo.networks.make_ppo_networks

make_inference_fn, params, metrics = brax.training.agents.ppo.train.train(
    environment=env,
    network_factory = brax.training.agents.ppo.networks.make_ppo_networks,
    wrap_env_fn=mujoco_playground.wrapper.wrap_for_brax_training,
    progress_fn=progress,
    **dict(ppo_params),
)

df = pd.DataFrame(dict(times=times, steps=x_data, reward=y_data, reward_std=y_dataerr, impl="brax"))
df.to_csv(f"benchmark_results/dm_control_suite/{env_name}_brax.csv")
'''

print("NNX-PPO")
x_data, y_data, y_dataerr = [np.nan], [np.nan], [np.nan]
times = [datetime.now()]
SEED = 1234
nnx_ppo_conf = config_dict.create(
    n_envs = ppo_params.num_envs,
    rollout_length = ppo_params.unroll_length,
    n_steps = ppo_params.num_timesteps,
    gae_lambda = 0.95,
    discounting_factor = ppo_params.discounting,
    clip_range = 0.3,
    normalize_advantages = True,
    normalize_observations = ppo_params.normalize_observations,
    n_epochs = ppo_params.num_updates_per_batch,
    episode_length = ppo_params.episode_length,
)

# Better params?
nnx_ppo_conf.entropy_weight = 1e-3
nnx_ppo_conf.learning_rate = 1e-4
nnx_ppo_conf.n_epochs = 4

train_env = reward_scaling_wrapper.RewardScalingWrapper(env, ppo_params.reward_scaling)
rngs = nnx.Rngs(SEED)
nets = MLPActorCritic(train_env.observation_size, train_env.action_size,
                      actor_hidden_sizes=[32,] * 4,
                      critic_hidden_sizes=[256,] * 5,
                      rngs=rngs,
                      transfer_function=nnx.swish,
                      action_sampler=NormalTanhSampler(rngs, entropy_weight=nnx_ppo_conf.entropy_weight,
                                                       min_std=1e-3, std_scale=1.0),
                      normalize_obs=nnx_ppo_conf.normalize_observations)
training_state = ppo.new_training_state(train_env, nets, n_envs=nnx_ppo_conf.n_envs,
                                        learning_rate=nnx_ppo_conf.learning_rate, seed=SEED)
ppo_step_jit = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3, 7, 8, 9))
eval_rollout_jit = nnx.jit(rollout.eval_rollout, static_argnums=(0, 2, 3))
last_eval = -ppo_params.num_timesteps
while training_state.steps_taken < ppo_params.num_timesteps:
    if training_state.steps_taken - last_eval > ppo_params.num_timesteps // ppo_params.num_evals:
        nets.eval() # Set network to eval mode
        eval_metrics = eval_rollout_jit(env, nets, 256, nnx_ppo_conf.episode_length, jax.random.key(SEED))
        times.append(datetime.now())
        x_data.append(training_state.steps_taken)
        y_data.append(eval_metrics["episode_reward_mean"])
        y_dataerr.append(eval_metrics["episode_reward_std"])
        print(f"{times[-1]} ({training_state.steps_taken}): {y_data[-1]}")
        nets.train() # Set the network back to train mode
        last_eval = training_state.steps_taken

    new_training_state, metrics = ppo_step_jit(
        train_env, training_state, 
        nnx_ppo_conf.n_envs, nnx_ppo_conf.rollout_length,
        nnx_ppo_conf.gae_lambda, nnx_ppo_conf.discounting_factor,
        nnx_ppo_conf.clip_range, nnx_ppo_conf.normalize_advantages,
        nnx_ppo_conf.n_epochs, ppo.LoggingLevel.NONE
    )
    training_state = new_training_state
    
df = pd.DataFrame(dict(times=times, steps=x_data, reward=y_data, reward_std=y_dataerr, impl="nnx-ppo"))
df.to_csv(f"benchmark_results/dm_control_suite/{env_name}_nnx_ppo.csv")
