import mujoco_playground
import jax
from flax import nnx

from nnx_ppo.networks.modules import MLPActorCritic
from nnx_ppo.algorithms import ppo, rollout
import nnx_ppo.test_dummies.parrot_env
import nnx_ppo.wrappers.episode_wrapper

SEED = 42
env = nnx_ppo.test_dummies.parrot_env.ParrotEnv(reward_falloff=0.5)
env = nnx_ppo.wrappers.episode_wrapper.EpisodeWrapper(env, max_len=100)
nets = MLPActorCritic(env.observation_size, env.action_size,
                      actor_hidden_sizes=[64, 64],
                      critic_hidden_sizes=[64, 64],
                      rngs=nnx.Rngs(SEED),
                      entropy_weight=1)
config = ppo.default_config()
config.discounting_factor = 0.0
training_state = ppo.new_training_state(env, nets, n_envs=config.n_envs, seed=SEED)
ppo_step_jit = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3))
eval_rollout_jit = nnx.jit(rollout.eval_rollout, static_argnums=(0, 2, 3))
for iter in range(5000):
    training_state, _ = ppo_step_jit(env, training_state, 
                                  config.n_envs, config.rollout_length,
                                  config.gae_lambda, config.discounting_factor,
                                  config.clip_range)
    if iter % 10 == 0:
        nets.eval() # Set network to eval mode
        eval_metrics = eval_rollout_jit(env, nets, 256, 100, jax.random.key(SEED))
        print(iter, training_state.steps_taken, "{:.4}".format(eval_metrics["episode_reward_mean"]))
        nets.train() # Set the network back to train mode