

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from datetime import datetime


import jax
import jax.numpy as jp
import numpy as np
from flax import nnx
import wandb
from ml_collections import config_dict

from vnl_playground.tasks.rodent.imitation import Imitation, default_config
from vnl_playground.tasks.wrappers import FlattenObsWrapper

from nnx_ppo.networks.modules import MLPActorCritic
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.algorithms import ppo, rollout


SEED = 42
env_config = default_config()
env_config.solver = "newton"
env_config.reward_terms["bodies_pos"]["weight"] = 0.0
env_config.reward_terms["joints_vel"]["weight"] = 0.0
env_config.mujoco_impl = "warp"
#env_config.nconmax = 256
env_config.naconmax = 2048
env_config.njmax = 256

net_config = config_dict.create(
    actor_hidden_sizes = [1024,] * 4,
    critic_hidden_sizes = [1024,] * 2,
    transfer_function = "swish",
    entropy_weight = 3e-3,
    min_std = 1e-2,
    std_scale = 1.0,
    preclamp_sampling = False,
    normalize_obs = True,
    initalizer_scale = 1.0,
)

train_env = FlattenObsWrapper(Imitation(env_config))
eval_env = train_env


rngs = nnx.Rngs(SEED)
transfer_function = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[net_config.transfer_function]
nets = MLPActorCritic(train_env.observation_size, train_env.action_size,
                      actor_hidden_sizes=net_config.actor_hidden_sizes,
                      critic_hidden_sizes=net_config.critic_hidden_sizes,
                      rngs=rngs,
                      transfer_function=transfer_function,
                      action_sampler=NormalTanhSampler(rngs,
                                                       entropy_weight=net_config.entropy_weight,
                                                       min_std=net_config.min_std,
                                                       std_scale=net_config.std_scale,
                                                       preclamp=net_config.preclamp_sampling),
                      normalize_obs=net_config.normalize_obs,
                      initalizer_scale=net_config.initalizer_scale)
config = ppo.default_config()
config.normalize_advantages = True
config.discounting_factor = 0.95
config.n_envs = 256
config.rollout_length = 20
config.learning_rate = 1e-4
config.n_epochs = 4
config.n_minibatches = 2
config.gradient_clipping = None
config.weight_decay = None

now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"SimpleMLP-{timestamp}"
wandb.init(project="nnx-ppo-rodent-imitation",
           config={"env": "StandardImitation",
                   "SEED": SEED,
                   "ppo_params": config.to_dict(),
                   "net_params": net_config.to_dict(),
                   "env_params": env_config.to_dict()},
           name=exp_name,
           tags=(["MLP"]),
           notes="Trying warp locally again.")

training_state_init = nnx.jit(ppo.new_training_state, static_argnums=[0,2,4,5,6])
training_state = training_state_init(train_env, nets, config.n_envs, SEED, config.learning_rate, config.gradient_clipping,
                                     config.weight_decay)

ppo_step_jit = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3, 7, 8, 9, 10, 11))
eval_rollout_jit = nnx.jit(rollout.eval_rollout, static_argnums=(0, 2, 3))
RENDER_EPISODE_LENGTH = 1000
eval_rollout_render_jit = nnx.jit(rollout.eval_rollout_for_render_scan, static_argnums=(0, 2))

nets.eval() # Set network to eval mode
eval_metrics = eval_rollout_jit(eval_env, nets, 256, 500, jax.random.key(SEED))
wandb.log({**eval_metrics, "n_steps": training_state.steps_taken})
nets.train() # Set the network back to train mode

for iter in range(200_000):
    new_training_state, metrics = ppo_step_jit(
        train_env, training_state, 
        config.n_envs, config.rollout_length,
        config.gae_lambda, config.discounting_factor,
        config.clip_range, config.normalize_advantages,
        config.n_epochs, config.n_minibatches, ppo.LoggingLevel.ALL,
        (0, 25, 50, 75, 100)
    )
    training_state = new_training_state
    if iter % 100 == 0:
        print(f"Iter {iter}: starting eval")
        nets.eval() # Set network to eval mode
        eval_metrics = eval_rollout_jit(eval_env, nets, 256, 500, jax.random.key(SEED))
        metrics.update(eval_metrics)
        nets.train() # Set the network back to train mode

    # Log rendered eval rollout video every 1000 iterations
    if iter % 1000 == 0 and hasattr(eval_env, 'render'):
        print(f"Iter {iter}: starting render eval")
        nets.eval()
        render_key = jax.random.fold_in(jax.random.key(SEED), iter)
        stacked_states, final_state, episode_reward = eval_rollout_render_jit(
            eval_env, nets, RENDER_EPISODE_LENGTH, render_key
        )
        trajectory = rollout.unstack_trajectory(stacked_states, final_state, RENDER_EPISODE_LENGTH)
        frames = eval_env.render(trajectory, height=480, width=640, camera="close_profile-rodent", add_labels=True)
        # Stack frames: (T, H, W, C) -> (T, C, H, W) for wandb
        video_array = np.stack(frames).transpose(0, 3, 1, 2)
        metrics["eval_video"] = wandb.Video(video_array, fps=50, format="mp4")
        nets.train()
    wandb.log(metrics)
