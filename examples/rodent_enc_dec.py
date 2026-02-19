"""Rodent imitation learning with encoder-decoder architecture and variational bottleneck."""
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

from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.containers import PPOActorCritic, Sequential, Concat, Flattener
from nnx_ppo.networks.factories import make_mlp, make_mlp_layers
from nnx_ppo.networks.variational import AR1VariationalBottleneck
from nnx_ppo.networks.normalizer import Normalizer
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
env_config.naconmax = 16 * 512
env_config.njmax = 256

net_config = config_dict.create(
    enc_hidden_sizes=[512] * 4,
    dec_hidden_sizes=[512] * 4,
    critic_hidden_sizes=[1024] * 2,
    activation="swish",
    entropy_weight=1e-2,
    min_std=1e-1,
    std_scale=1.0,
    normalize_obs=True,
    initalizer_scale=1.0,
    kl_weight=0.01,
    latent_min_std=0.01,
    latent_size=32,
)

config = TrainConfig(
    ppo=PPOConfig(
        n_envs=512,
        rollout_length=20,
        total_steps=1_000_000_000,
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
        every_steps=10_000_000,
        n_envs=512,
        max_episode_length=500,
        logging_percentiles=(0, 25, 50, 75, 100),
    ),
    video=VideoConfig(
        enabled=True,
        every_steps=50_000_000,
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

train_env = Imitation(env_config)
eval_env = train_env
obs_size = train_env.non_flattened_observation_size

# Build encoder-decoder network
rngs = nnx.Rngs(SEED)
activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[net_config.activation]
reference_size = int(sum(jax.tree.flatten(obs_size["imitation_target"])[0]))
proprio_size = int(sum(jax.tree.flatten(obs_size["proprioception"])[0]))
enc_sizes = [reference_size] + net_config.enc_hidden_sizes + [net_config.latent_size * 2]
dec_sizes = [net_config.latent_size + proprio_size] + net_config.dec_hidden_sizes + [train_env.action_size * 2]

actor = Sequential([
    Concat(
        imitation_target=Sequential([
            Flattener(),
            *make_mlp_layers(enc_sizes, rngs, activation, activation_last_layer=False),
            VariationalBottleneck(net_config.latent_size, rngs, net_config.kl_weight, net_config.latent_min_std),
        ]),
        proprioception=Flattener(),
    ),
    make_mlp(dec_sizes, rngs, activation, activation_last_layer=False)
])
critic_sizes = [reference_size + proprio_size] + net_config.critic_hidden_sizes + [1]
critic = Sequential([
    Flattener(),
    *make_mlp_layers(critic_sizes, rngs, activation, activation_last_layer=False),
])
sampler = NormalTanhSampler(
    rngs,
    entropy_weight=net_config.entropy_weight,
    min_std=net_config.min_std,
    std_scale=net_config.std_scale
)
nets = PPOActorCritic(
    preprocessor=Normalizer(obs_size),
    actor=actor,
    critic=critic,
    action_sampler=sampler,
)



# Initialize wandb
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"EncDec-{timestamp}"
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
    tags=("MLP", "warp", "EncDec"),
    notes="Testing AR1 bottleneck",
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
if result.eval_history:
    print(f"Final eval reward: {result.eval_history[-1].get('episode_reward_mean', 'N/A')}")
