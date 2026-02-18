"""Rodent imitation learning with simple MLP architecture."""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from datetime import datetime
import dataclasses

import jax
import numpy as np
from flax import nnx
import wandb
from ml_collections import config_dict

from vnl_playground.tasks.rodent.imitation import Imitation, default_config
from vnl_playground.tasks.wrappers import FlattenObsWrapper

from nnx_ppo.networks.feedforward import MLPActorCritic
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.algorithms.config import (
    TrainConfig, PPOConfig, EvalConfig, VideoConfig
)
from nnx_ppo.algorithms.callbacks import wandb_video_fn


SEED = 40
env_config = default_config()
env_config.solver = "newton"
env_config.reward_terms["bodies_pos"]["weight"] = 0.0
env_config.reward_terms["joints_vel"]["weight"] = 0.0
env_config.mujoco_impl = "warp"
env_config.naconmax = 32 * 2048
env_config.njmax = 256

net_config = config_dict.create(
    actor_hidden_sizes=[1024] * 4,
    critic_hidden_sizes=[1024] * 2,
    transfer_function="swish",
    entropy_weight=1e-2,
    min_std=1e-1,
    std_scale=1.0,
    normalize_obs=True,
    initalizer_scale=1.0,
)

train_env = FlattenObsWrapper(Imitation(env_config))
eval_env = train_env

# Setup network
rngs = nnx.Rngs(SEED)
transfer_function = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[net_config.transfer_function]
nets = MLPActorCritic(
    train_env.observation_size,
    train_env.action_size,
    actor_hidden_sizes=net_config.actor_hidden_sizes,
    critic_hidden_sizes=net_config.critic_hidden_sizes,
    rngs=rngs,
    transfer_function=transfer_function,
    action_sampler=NormalTanhSampler(
        rngs,
        entropy_weight=net_config.entropy_weight,
        min_std=net_config.min_std,
        std_scale=net_config.std_scale,
    ),
    normalize_obs=net_config.normalize_obs,
    initalizer_scale=net_config.initalizer_scale
)

# Setup config using new dataclass API
config = TrainConfig(
    ppo=PPOConfig(
        n_envs=2048,
        rollout_length=20,
        total_steps=200_000 * 2048 * 20,  # 200k iterations
        discounting_factor=0.95,
        normalize_advantages=True,
        learning_rate=1e-4,
        n_epochs=4,
        n_minibatches=8,
        gradient_clipping=1.0,
        weight_decay=None,
        logging_level=LoggingLevel.ALL,
        logging_percentiles=(0, 25, 50, 75, 100),
    ),
    eval=EvalConfig(
        enabled=True,
        every_steps=100 * 2048 * 20,  # Every 100 iterations
        n_envs=256,
        max_episode_length=500,
        logging_percentiles=(0, 25, 50, 75, 100),
    ),
    video=VideoConfig(
        enabled=True,
        every_steps=1000 * 2048 * 20,  # Every 1000 iterations
        episode_length=1000,
        render_kwargs={
            "height": 480,
            "width": 640,
            "camera": "close_profile-rodent",
            "add_labels": True,
        },
    ),
    seed=SEED,
)

# Initialize wandb
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"SimpleMLP-{timestamp}"
wandb.init(
    project="nnx-ppo-rodent-imitation",
    config={
        "env": "StandardImitation",
        "SEED": SEED,
        "config": dataclasses.asdict(config),
        "net_params": net_config.to_dict(),
        "env_params": env_config.to_dict(),
    },
    name=exp_name,
    tags=("MLP", "warp"),
    notes="Using new train_ppo API",
)

# Train with wandb callbacks
result = ppo.train_ppo(
    train_env,
    nets,
    config,
    log_fn=wandb.log,
    video_fn=wandb_video_fn(fps=50),
    eval_env=eval_env,
)

print(f"Training complete: {result.total_steps} steps, {result.total_iterations} iterations")
print(f"Final eval reward: {result.eval_history[-1].get('episode_reward_mean', 'N/A')}")
