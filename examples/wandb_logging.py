from datetime import datetime

import mujoco_playground
import jax
from flax import nnx
import wandb

from nnx_ppo.networks.modules import MLPActorCritic
from nnx_ppo.algorithms import ppo, rollout

from nnx_ppo.test_dummies import episode_wrapper
SEED = 42
env_name = "CartpoleBalance"

env = mujoco_playground.registry.load(env_name)
train_env = episode_wrapper.EpisodeWrapper(env, 1000)
eval_env = train_env#env

nets = MLPActorCritic(env.observation_size, env.action_size,
                      actor_hidden_sizes=[64, 64],
                      critic_hidden_sizes=[64, 64],
                      rngs=nnx.Rngs(SEED),
                      transfer_function=nnx.tanh,
                      entropy_weight=0.0)
config = ppo.default_config()
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"{env_name}-{timestamp}"
wandb.init(project="nnx-ppo-basic-tests",
           config={"env": env_name,
                   "SEED": SEED,
                   "ppo_params": config.to_dict()},
           name=exp_name,
           tags=(env_name,))

training_state = ppo.new_training_state(env, nets, n_envs=config.n_envs, learning_rate=1e-4, seed=SEED)
ppo_step_jit = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3))
eval_rollout_jit = nnx.jit(rollout.eval_rollout, static_argnums=(0, 2, 3))
for iter in range(1000):
    training_state, metrics = ppo_step_jit(
        train_env, training_state, 
        config.n_envs, config.rollout_length,
        config.gae_lambda, config.discounting_factor,
        config.clip_range
    )
    if iter % 10 == 0:
        nets.eval() # Set network to eval mode
        eval_metrics = eval_rollout_jit(eval_env, nets, 256, 1000, jax.random.key(SEED))
        metrics.update(eval_metrics)
        metrics["n_steps"] = training_state.steps_taken
        wandb.log(metrics)
        #print(iter, training_state.steps_taken, "{:.4}".format(eval_metrics["episode_reward_mean"]))
        nets.train() # Set the network back to train mode