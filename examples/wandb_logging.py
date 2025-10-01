from datetime import datetime

import mujoco_playground
import jax
import jax.numpy as jp
from flax import nnx
import wandb

from nnx_ppo.networks.modules import MLPActorCritic
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.algorithms import ppo, rollout

from nnx_ppo.wrappers import episode_wrapper
import nnx_ppo.test_dummies.parrot_env
import nnx_ppo.test_dummies.move_to_center_env
import nnx_ppo.test_dummies.move_from_center_env
import nnx_ppo.test_dummies.action_sigma_debug as action_sigma_debug

jax.config.update("jax_debug_nans", True)

SEED = 43
env_name = "CartpoleBalance"

if env_name == "ParrotEnv":
    env = nnx_ppo.test_dummies.parrot_env.ParrotEnv(reward_falloff=1.0)
elif env_name == "MoveToCenterEnv":
    env = nnx_ppo.test_dummies.move_to_center_env.MoveToCenterEnv(reward_falloff=1.0, border_radius=4.0)
elif env_name == "MoveFromCenterEnv":
    env = nnx_ppo.test_dummies.move_from_center_env.MoveFromCenterEnv(border_radius=10.0)
else:
    env = mujoco_playground.registry.load(env_name)
train_env = episode_wrapper.EpisodeWrapper(env, 50)
eval_env = env

nets = MLPActorCritic(env.observation_size, env.action_size,
                      actor_hidden_sizes=[64, 64],
                      critic_hidden_sizes=[64, 64],
                      rngs=nnx.Rngs(SEED),
                      transfer_function=nnx.softplus,
                      action_sampler=NormalTanhSampler(entropy_weight=0.0, min_std=1e-1, std_scale=0.1))
config = ppo.default_config()
#config.discounting_factor = 0.0
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"{env_name}-{timestamp}"
wandb.init(project="nnx-ppo-basic-tests",
           config={"env": env_name,
                   "SEED": SEED,
                   "ppo_params": config.to_dict()},
           name=exp_name,
           tags=(env_name,))

training_state = ppo.new_training_state(train_env, nets, n_envs=config.n_envs, learning_rate=1e-4, seed=SEED)
ppo_step_jit = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3, 7))
eval_rollout_jit = nnx.jit(rollout.eval_rollout, static_argnums=(0, 2, 3))

nets.eval() # Set network to eval mode
eval_metrics = eval_rollout_jit(eval_env, nets, 256, 100, jax.random.key(SEED))
wandb.log({**eval_metrics, "n_steps": training_state.steps_taken})
nets.train() # Set the network back to train mode

@nnx.jit(static_argnums=(0, 2, 3))
def extra_logging(env, training_state, n_envs: int, rollout_length: int, reset_keys):
    unroll_vmap = nnx.vmap(rollout.unroll_env,
                           in_axes  = (None, 0, None, 0, None, 0),
                           out_axes = (0, 0, 0))
    unroll_vmap = nnx.split_rngs(splits=n_envs)(unroll_vmap)
    next_net_state, next_env_state, rollout_data = unroll_vmap(
        env,
        training_state.env_states,
        training_state.networks,
        training_state.network_states,
        rollout_length,
        reset_keys
    )
    return action_sigma_debug.extra_metrics(training_state.networks, rollout_data)

for iter in range(5000):
    training_state, metrics = ppo_step_jit(
        train_env, training_state, 
        config.n_envs, config.rollout_length,
        config.gae_lambda, config.discounting_factor,
        config.clip_range, config.normalize_advantages
    )
    if iter % 1 == 0:
        metrics.update(extra_logging(train_env, training_state, 256, config.rollout_length,
                                     jax.random.split(jax.random.key(SEED), 256)))
        nets.eval() # Set network to eval mode
        eval_metrics = eval_rollout_jit(eval_env, nets, 256, 100, jax.random.key(SEED))
        metrics.update(eval_metrics)
        metrics["n_steps"] = training_state.steps_taken
        for k, v in metrics.items():
            if jp.any(jp.isnan(v)):
                raise ValueError(f"NaN in {k}")
        wandb.log(metrics)
        #print(iter, training_state.steps_taken, "{:.4}".format(eval_metrics["episode_reward_mean"]))
        nets.train() # Set the network back to train mode