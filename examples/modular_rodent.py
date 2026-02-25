"""Rodent imitation learning with simple MLP architecture."""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from datetime import datetime
import dataclasses

import jax.numpy as jp
import jax.flatten_util
from flax import nnx
import wandb
from ml_collections import config_dict

from vnl_playground.tasks.modular_rodent.imitation import ModularImitation, default_config

from nnx_ppo.networks.factories import make_mlp_actor_critic
from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.types import LoggingLevel, RLEnv, EnvState
from nnx_ppo.algorithms.config import TrainConfig, PPOConfig, EvalConfig, VideoConfig
from nnx_ppo.algorithms.callbacks import wandb_video_fn

SEED = 40
env_config = default_config()
#env_config.solver = "newton"
#env_config.reward_terms["bodies_pos"]["weight"] = 0.0
#env_config.reward_terms["joints_vel"]["weight"] = 0.0
#env_config.mujoco_impl = "warp"
env_config.naconmax = 64*1024
env_config.njmax = 256
#env_config.naconmax = 32 * 2048
#env_config.njmax = 256

net_config = config_dict.create(
    actor_hidden_sizes=[1024] * 4,
    critic_hidden_sizes=[1024] * 2,
    activation="swish",
    entropy_weight=1e-2,
    min_std=1e-1,
    std_scale=1.0,
    normalize_obs=True,
    initializer_scale=1.0,
)

config = TrainConfig(
    ppo=PPOConfig(
        n_envs=1024,
        rollout_length=20,
        total_steps=500_000_000,
        discounting_factor=0.95,
        normalize_advantages=True,
        learning_rate=1e-4,
        n_epochs=4,
        n_minibatches=1,
        gradient_clipping=1.0,
        weight_decay=None,
        logging_level=LoggingLevel.LOSSES | LoggingLevel.TRAIN_ROLLOUT_STATS,
        logging_percentiles=(0, 25, 50, 75, 100),
    ),
    eval=EvalConfig(
        enabled=True,
        every_steps=5_000_000,
        n_envs=512,
        max_episode_length=500,
        logging_percentiles=(0, 25, 50, 75, 100),
    ),
    video=VideoConfig(
        enabled=True,
        every_steps=10_000_000,
        episode_length=1000,
        render_kwargs={
            "height": 480,
            "width": 640,
            "camera": "close_profile",
            "add_labels": True,
        },
    ),
    seed=SEED,
)

class CustomFlattenWrapper(RLEnv):
    def __init__(self, base_env: RLEnv) -> None:
        self.base_env = base_env
        null_action = self.base_env.null_action()
        _, self.unwrap_action = jax.flatten_util.ravel_pytree(null_action)

    def reset(self, rng):
        state = self.base_env.reset(rng)
        return self._flatten(state)

    def step(self, state, action: jax.Array) -> EnvState:
        dict_action = self.unwrap_action(action)
        new_state = self.base_env.step(state, dict_action)
        return self._flatten(new_state)

    def _flatten(self, state: EnvState) -> EnvState:
        new_obs, _ = jax.flatten_util.ravel_pytree(state.obs)
        new_reward = jax.tree.reduce(jp.add, state.reward)
        new_state = state.replace(
            obs = new_obs,
            reward = new_reward,
        )
        return new_state
    
    @property
    def observation_size(self) -> jax.Array:
        return jax.tree.reduce(jp.add, self.base_env.observation_size)

    @property
    def action_size(self) -> jax.Array:
        return jax.tree.reduce(jp.add, self.base_env.action_size)

    def render(self, trajectory, **kwargs):
        return self.base_env.render(trajectory, **kwargs)    

base_env = ModularImitation(env_config)
train_env = CustomFlattenWrapper(base_env)
eval_env = train_env

# Setup network
rngs = nnx.Rngs(SEED)

nets = make_mlp_actor_critic(
    train_env.observation_size, train_env.action_size, rngs=rngs, **net_config
)

# Initialize wandb
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"Modular-{timestamp}"
wandb.init(
    project="nnx-ppo-rodent-imitation",
    config={
        "env": "ModularImitation",
        "SEED": SEED,
        "config": dataclasses.asdict(config),
        "net_params": net_config.to_dict(),
        "env_params": env_config.to_dict(),
    },
    name=exp_name,
    tags=("MLP", "warp", "Modular"),
    notes="Trying modular rodent env locally.",
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

print(
    f"Training complete: {result.total_steps} steps, {result.total_iterations} iterations"
)
print(f"Final eval reward: {result.eval_history[-1].get('episode_reward_mean', 'N/A')}")
