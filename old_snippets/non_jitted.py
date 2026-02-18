import mujoco_playground
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.feedforward import MLPActorCritic
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.algorithms import ppo, rollout

from nnx_ppo.wrappers import episode_wrapper
import nnx_ppo.test_dummies.parrot_env
import nnx_ppo.test_dummies.move_to_center_env
import nnx_ppo.test_dummies.move_from_center_env
import nnx_ppo.test_dummies.action_sigma_debug as action_sigma_debug

#jax.config.update("jax_debug_nans", True)

SEED = 39
env_name = "ReacherEasy"#"CartpoleBalance"

if env_name == "ParrotEnv":
    env = nnx_ppo.test_dummies.parrot_env.ParrotEnv(reward_falloff=1.0)
elif env_name == "MoveToCenterEnv":
    env = nnx_ppo.test_dummies.move_to_center_env.MoveToCenterEnv(reward_falloff=1.0, border_radius=10.0)
elif env_name == "MoveFromCenterEnv":
    env = nnx_ppo.test_dummies.move_from_center_env.MoveFromCenterEnv(border_radius=10.0)
else:
    env = mujoco_playground.registry.load(env_name)
train_env = env#episode_wrapper.EpisodeWrapper(env, 100)
eval_env = env

rngs = nnx.Rngs(SEED)
nets = MLPActorCritic(env.observation_size, env.action_size,
                      actor_hidden_sizes=[64,] * 4,
                      critic_hidden_sizes=[64,] * 4,
                      rngs=rngs,
                      transfer_function=nnx.tanh,
                      action_sampler=NormalTanhSampler(rngs, entropy_weight=5e-3, min_std=1e-2, std_scale=1.0, preclamp=False),
                      normalize_obs=True)
config = ppo.default_config()
config.normalize_advantages = False
config.discounting_factor = 0.99#95#95
config.n_envs = 256
config.rollout_length = 20

training_state = ppo.new_training_state(train_env, nets, n_envs=config.n_envs, learning_rate=1e-4, seed=SEED)

nets.eval() # Set network to eval mode
eval_metrics = rollout.eval_rollout(eval_env, nets, 64, 100, jax.random.key(SEED))
nets.train() # Set the network back to train mode

for iter in range(5000):
    new_training_state, metrics = ppo.ppo_step(
        train_env, training_state, 
        config.n_envs, config.rollout_length,
        config.gae_lambda, config.discounting_factor,
        config.clip_range, config.normalize_advantages,
        ppo.LoggingLevel.ALL
    )
    metrics.update(extra_logging(train_env, training_state, config.rollout_length))
    training_state = new_training_state
    if iter % 10 == 0:
        nets.eval() # Set network to eval mode
        eval_metrics = eval_rollout_jit(eval_env, nets, 64, 100, jax.random.key(SEED))
        metrics.update(eval_metrics)
        metrics["n_steps"] = training_state.steps_taken
        nets.train() # Set the network back to train mode
