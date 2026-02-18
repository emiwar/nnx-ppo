
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
config = ppo.default_config()
config.normalize_advantages = True
config.discounting_factor = 0.99
config.n_envs = 256
config.rollout_length = 30
config.n_epochs = 4
config.n_minibatches = 4

training_state = ppo.new_training_state(train_env, nets, n_envs=config.n_envs, learning_rate=1e-4, seed=SEED)

for iter in range(5):
    new_training_state, metrics = ppo.ppo_step(
        train_env, training_state, 
        config.n_envs, config.rollout_length,
        config.gae_lambda, config.discounting_factor,
        config.clip_range, config.normalize_advantages,
        config.n_epochs, config.n_minibatches, ppo.LoggingLevel.ALL,
        (0, 25, 50, 75, 100)
    )

    def_1 = jax.tree.structure(training_state)
    def_2 = jax.tree.structure(new_training_state)
    if def_1 != def_2:
        raise AssertionError(f"ppo_step changed return type from {def_1} to {def_2}")

    training_state = new_training_state
    print(metrics)

    if iter % 100 == 0:
        print(f"Iter {iter}: starting eval")
        nets.eval() # Set network to eval mode
        eval_metrics = rollout.eval_rollout(eval_env, nets, 128, 500, jax.random.key(SEED))
        metrics.update(eval_metrics)
        nets.train() # Set the network back to train mode

    # Log rendered eval rollout video every 500 iterations
    if iter % 1000 == 0 and hasattr(eval_env, 'render'):
        print(f"Iter {iter}: starting render eval")
        nets.eval()
        render_key = jax.random.fold_in(jax.random.key(SEED), iter)
        stacked_states, final_state, episode_reward = rollout.eval_rollout_for_render_scan(
            eval_env, nets, 1000, render_key
        )
        trajectory = rollout.unstack_trajectory(stacked_states, final_state, RENDER_EPISODE_LENGTH)
        frames = eval_env.render(trajectory, height=480, width=640, add_labels=True)
        # Stack frames: (T, H, W, C) -> (T, C, H, W) for wandb
        video_array = np.stack(frames).transpose(0, 3, 1, 2)
        metrics["eval_video"] = wandb.Video(video_array, fps=50, format="mp4")
        nets.train()
    wandb.log(metrics)