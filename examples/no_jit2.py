from datetime import datetime

import tqdm
import mujoco_playground
import jax
from flax import nnx
import wandb

from nnx_ppo.networks.modules import MLPActorCritic
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.algorithms import ppo, rollout

from nnx_ppo.wrappers import episode_wrapper
import nnx_ppo.test_dummies.parrot_env
import nnx_ppo.test_dummies.move_to_center_env

SEED = 64
env_name = "MoveToCenterEnv"#"CartpoleBalance"

if env_name == "ParrotEnv":
    env = nnx_ppo.test_dummies.parrot_env.ParrotEnv(reward_falloff=1.0)
elif env_name == "MoveToCenterEnv":
    env = nnx_ppo.test_dummies.move_to_center_env.MoveToCenterEnv(reward_falloff=1.0, border_radius=4.0)
else:
    env = mujoco_playground.registry.load(env_name)
train_env = env #episode_wrapper.EpisodeWrapper(env, 1000)
eval_env = env

nets = MLPActorCritic(env.observation_size, env.action_size,
                      actor_hidden_sizes=[64, 64],
                      critic_hidden_sizes=[64, 64],
                      rngs=nnx.Rngs(SEED),
                      transfer_function=nnx.tanh,
                      action_sampler=NormalTanhSampler(entropy_weight=0.1))
config = ppo.default_config()
#config.discounting_factor = 0.0

training_state = ppo.new_training_state(train_env, nets, n_envs=config.n_envs, learning_rate=1e-4, seed=SEED)
ppo_step_jit = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3))
eval_rollout_jit = nnx.jit(rollout.eval_rollout, static_argnums=(0, 2, 3))

for iter in tqdm.trange(200):
    training_state, metrics = ppo_step_jit(
        train_env, training_state, 
        config.n_envs, config.rollout_length,
        config.gae_lambda, config.discounting_factor,
        config.clip_range
    )
for iter in tqdm.trange(200):
    training_state, metrics = ppo.ppo_step(
            train_env, training_state, 
            config.n_envs, config.rollout_length,
            config.gae_lambda, config.discounting_factor,
            config.clip_range
        )