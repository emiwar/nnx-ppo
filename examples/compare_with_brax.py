"""Benchmark comparison between nnx-ppo and brax PPO implementations."""

from datetime import datetime

import pandas as pd
import numpy as np

import mujoco_playground
import mujoco_playground.config.dm_control_suite_params

import brax
import brax.training.agents.ppo.train
import brax.training.agents.ppo.networks

from flax import nnx
from nnx_ppo.networks.factories import make_mlp_actor_critic
from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.algorithms.config import TrainConfig, PPOConfig, EvalConfig
from nnx_ppo.wrappers import reward_scaling_wrapper

env_name = "CartpoleBalance"

env = mujoco_playground.registry.load(env_name)
env_cfg = mujoco_playground.registry.get_default_config(env_name)
ppo_params = mujoco_playground.config.dm_control_suite_params.brax_ppo_config(env_name)
ppo_params.num_evals = 100

print("NNX-PPO")
x_data, y_data, y_dataerr = [np.nan], [np.nan], [np.nan]
times = [datetime.now()]
SEED = 1234

config = TrainConfig(
    ppo=PPOConfig(
        n_envs=ppo_params.num_envs,
        # Brax effectively rolls out batch_size * num_minibatches environments
        # of length unroll_length per training step, looping the actual
        # num_envs parallel envs (batch_size * num_minibatches // num_envs)
        # times. Match that by scaling rollout_length so each nnx-ppo
        # iteration consumes the same number of env steps before optimizing.
        rollout_length=ppo_params.unroll_length
        * (ppo_params.batch_size * ppo_params.num_minibatches // ppo_params.num_envs),
        total_steps=ppo_params.num_timesteps,
        gae_lambda=0.95,
        discounting_factor=ppo_params.discounting,
        clip_range=0.3,
        normalize_advantages=True,
        n_epochs=ppo_params.num_updates_per_batch,
        learning_rate=ppo_params.learning_rate,
        n_minibatches=ppo_params.num_minibatches,
        # Brax's value loss is 0.5 * 0.5 * MSE; nnx-ppo's is 0.5 * MSE * weight.
        # Use 0.5 to match brax's effective 0.25 * MSE.
        critic_loss_weight=0.5,
        logging_level=LoggingLevel.NONE,
    ),
    eval=EvalConfig(
        enabled=True,
        every_steps=ppo_params.num_timesteps // ppo_params.num_evals,
        n_envs=256,
        max_episode_length=ppo_params.episode_length,
        logging_percentiles=None,  # emit episode_reward/mean + /std (brax-style)
    ),
    seed=SEED,
)

train_env = reward_scaling_wrapper.RewardScalingWrapper(env, ppo_params.reward_scaling)
rngs = nnx.Rngs(SEED)
nets = make_mlp_actor_critic(
    train_env.observation_size,
    train_env.action_size,
    actor_hidden_sizes=[32] * 4,
    critic_hidden_sizes=[256] * 5,
    rngs=rngs,
    activation=nnx.swish,
    normalize_obs=ppo_params.normalize_observations,
    entropy_weight=ppo_params.entropy_cost,
    min_std=1e-3,
    std_scale=1.0,
)


def log_fn(metrics: dict, steps: int) -> None:
    # train_ppo merges eval metrics into the per-step metrics dict only on
    # the steps an eval runs. Use the presence of `episode_reward/mean` as
    # the signal that this step carried an eval result.
    if "episode_reward/mean" not in metrics:
        return
    times.append(datetime.now())
    x_data.append(steps)
    y_data.append(metrics["episode_reward/mean"])
    y_dataerr.append(metrics["episode_reward/std"])
    print(f"{times[-1]} ({steps}): {y_data[-1]}")


ppo.train_ppo(env=train_env, networks=nets, config=config, log_fn=log_fn, eval_env=env)

df = pd.DataFrame(
    dict(times=times, steps=x_data, reward=y_data, reward_std=y_dataerr, impl="nnx-ppo")
)
df.to_csv(f"benchmark_results/dm_control_suite/{env_name}_nnx_ppo.csv")


print("BRAX")
x_data, y_data, y_dataerr = [np.nan], [np.nan], [np.nan]
times = [datetime.now()]

def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])
    print(f"{times[-1]} ({num_steps}): {y_data[-1]}")


make_inference_fn, params, metrics = brax.training.agents.ppo.train.train(
    environment=env,
    network_factory=brax.training.agents.ppo.networks.make_ppo_networks,
    wrap_env_fn=mujoco_playground.wrapper.wrap_for_brax_training,
    progress_fn=progress,
    **dict(ppo_params),
)

df = pd.DataFrame(
    dict(times=times, steps=x_data, reward=y_data, reward_std=y_dataerr, impl="brax")
)
df.to_csv(f"benchmark_results/dm_control_suite/{env_name}_brax.csv")