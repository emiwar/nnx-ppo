"""Benchmark comparison between nnx-ppo and brax PPO implementations."""
from datetime import datetime
import functools

import pandas as pd
import numpy as np
import jax

import mujoco_playground
import mujoco_playground.config.dm_control_suite_params

import brax
import brax.training.agents.ppo.train
import brax.training.agents.ppo.networks

from flax import nnx
from nnx_ppo.networks.factories import make_mlp_actor_critic
from nnx_ppo.algorithms import ppo, rollout
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.algorithms.config import TrainConfig, PPOConfig, EvalConfig
from nnx_ppo.wrappers import reward_scaling_wrapper

env_name = 'WalkerStand'

env = mujoco_playground.registry.load(env_name)
env_cfg = mujoco_playground.registry.get_default_config(env_name)
ppo_params = mujoco_playground.config.dm_control_suite_params.brax_ppo_config(env_name)
ppo_params.num_evals = 100

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

print("NNX-PPO")
x_data, y_data, y_dataerr = [np.nan], [np.nan], [np.nan]
times = [datetime.now()]
SEED = 1234

# Configure using new dataclass API
config = TrainConfig(
    ppo=PPOConfig(
        n_envs=ppo_params.num_envs,
        rollout_length=ppo_params.unroll_length,
        total_steps=ppo_params.num_timesteps,
        gae_lambda=0.95,
        discounting_factor=ppo_params.discounting,
        clip_range=0.3,
        normalize_advantages=True,
        n_epochs=ppo_params.num_updates_per_batch,
        learning_rate=ppo_params.learning_rate,
        n_minibatches=ppo_params.num_minibatches,
        logging_level=LoggingLevel.NONE,
    ),
    eval=EvalConfig(
        enabled=False,  # Manual eval for timing comparison
        n_envs=256,
        max_episode_length=ppo_params.episode_length,
    ),
    seed=SEED,
)

train_env = reward_scaling_wrapper.RewardScalingWrapper(env, ppo_params.reward_scaling)
rngs = nnx.Rngs(SEED)
nets = make_mlp_actor_critic(
    train_env.observation_size, train_env.action_size,
    actor_hidden_sizes=[32] * 4,
    critic_hidden_sizes=[256] * 5,
    rngs=rngs,
    activation=nnx.swish,
    normalize_obs=ppo_params.normalize_observations,
    entropy_weight=ppo_params.entropy_cost,
    min_std=1e-3,
    std_scale=1.0,
)

# Manual training loop for precise timing comparison
training_state = ppo.new_training_state(
    train_env, nets, n_envs=config.ppo.n_envs,
    learning_rate=config.ppo.learning_rate, seed=SEED
)
ppo_step_jit = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3, 7, 8, 9, 10, 11))
eval_rollout_jit = nnx.jit(rollout.eval_rollout, static_argnums=(0, 2, 3))

eval_interval = config.ppo.total_steps // ppo_params.num_evals
last_eval = -eval_interval

while training_state.steps_taken < config.ppo.total_steps:
    if training_state.steps_taken - last_eval > eval_interval:
        nets.eval()
        eval_metrics = eval_rollout_jit(
            env, nets, config.eval.n_envs,
            config.eval.max_episode_length, jax.random.key(SEED)
        )
        times.append(datetime.now())
        x_data.append(training_state.steps_taken)
        y_data.append(eval_metrics["episode_reward_mean"])
        y_dataerr.append(eval_metrics["episode_reward_std"])
        print(f"{times[-1]} ({training_state.steps_taken}): {y_data[-1]}")
        nets.train()
        last_eval = training_state.steps_taken

    training_state, metrics = ppo_step_jit(
        train_env, training_state,
        config.ppo.n_envs, config.ppo.rollout_length,
        config.ppo.gae_lambda, config.ppo.discounting_factor,
        config.ppo.clip_range, config.ppo.normalize_advantages,
        config.ppo.n_epochs, config.ppo.n_minibatches,
        config.ppo.logging_level, config.ppo.logging_percentiles
    )

df = pd.DataFrame(dict(times=times, steps=x_data, reward=y_data, reward_std=y_dataerr, impl="nnx-ppo"))
df.to_csv(f"benchmark_results/dm_control_suite/{env_name}_nnx_ppo.csv")
