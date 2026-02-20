"""Example of using the new train_ppo API with wandb logging."""

from datetime import datetime
import dataclasses

import mujoco_playground
from flax import nnx
import wandb

from nnx_ppo.networks.factories import make_mlp_actor_critic
from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.algorithms.config import TrainConfig, PPOConfig, EvalConfig, VideoConfig
from nnx_ppo.algorithms.callbacks import wandb_video_fn

from nnx_ppo.wrappers import episode_wrapper
import nnx_ppo.test_dummies.parrot_env
import nnx_ppo.test_dummies.move_to_center_env
import nnx_ppo.test_dummies.move_from_center_env

SEED = 40
env_name = "CartpoleSwingup"

# Setup environment
if env_name == "ParrotEnv":
    env = nnx_ppo.test_dummies.parrot_env.ParrotEnv(reward_falloff=1.0)
elif env_name == "MoveToCenterEnv":
    env = nnx_ppo.test_dummies.move_to_center_env.MoveToCenterEnv(
        reward_falloff=1.0, border_radius=10.0
    )
elif env_name == "MoveFromCenterEnv":
    env = nnx_ppo.test_dummies.move_from_center_env.MoveFromCenterEnv(
        border_radius=10.0
    )
else:
    env = mujoco_playground.registry.load(env_name)

train_env = episode_wrapper.EpisodeWrapper(env, 1000)
eval_env = env

# Setup network
rngs = nnx.Rngs(SEED)
nets = make_mlp_actor_critic(
    env.observation_size,
    env.action_size,
    actor_hidden_sizes=[64] * 4,
    critic_hidden_sizes=[256] * 2,
    rngs=rngs,
    activation=nnx.swish,
    normalize_obs=True,
    entropy_weight=1e-2,
    min_std=5e-3,
    std_scale=1.0,
)

# Setup config using new dataclass API
config = TrainConfig(
    ppo=PPOConfig(
        n_envs=1024,
        rollout_length=30,
        total_steps=10_000 * 1024 * 30,  # ~300M steps
        discounting_factor=0.99,
        normalize_advantages=True,
        n_epochs=4,
        n_minibatches=4,
        logging_level=LoggingLevel.ALL,
        logging_percentiles=(0, 25, 50, 75, 100),
    ),
    eval=EvalConfig(
        enabled=True,
        every_steps=50 * 1024 * 30,  # Every ~1.5M steps
        n_envs=64,
        max_episode_length=1000,
        logging_percentiles=(0, 25, 50, 75, 100),
    ),
    video=VideoConfig(
        enabled=True,
        every_steps=500 * 1024 * 30,  # Every ~15M steps
        episode_length=1000,
        render_kwargs={"height": 240, "width": 320},
    ),
    seed=SEED,
)

# Initialize wandb
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"{env_name}-{timestamp}"
wandb.init(
    project="nnx-ppo-basic-tests",
    config={
        "env": env_name,
        "SEED": SEED,
        "config": dataclasses.asdict(config),
    },
    name=exp_name,
    tags=(env_name,),
    notes="Using new train_ppo API",
)

# Train with wandb callbacks
result = ppo.train_ppo(
    train_env,
    nets,
    config,
    log_fn=wandb.log,
    video_fn=wandb_video_fn(fps=30),
    eval_env=eval_env,
)

print(
    f"Training complete: {result.total_steps} steps, {result.total_iterations} iterations"
)
print(f"Final eval reward: {result.eval_history[-1].get('episode_reward_mean', 'N/A')}")
