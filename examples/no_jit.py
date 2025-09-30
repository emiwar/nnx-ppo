import mujoco_playground
import jax
from flax import nnx
import tqdm

from nnx_ppo.networks.modules import MLPActorCritic
from nnx_ppo.algorithms import ppo, rollout
from nnx_ppo.networks.sampling_layers import NormalSampler
import nnx_ppo.test_dummies.parrot_env
import nnx_ppo.test_dummies.move_to_center_env
import nnx_ppo.wrappers.episode_wrapper

SEED = 42
#env = nnx_ppo.test_dummies.parrot_env.ParrotEnv(reward_falloff=1.0)
#env = nnx_ppo.wrappers.episode_wrapper.EpisodeWrapper(env, max_len=100)
env = nnx_ppo.test_dummies.move_to_center_env.MoveToCenterEnv(reward_falloff=1.0, border_radius=4.0)
nets = MLPActorCritic(env.observation_size, env.action_size,
                      actor_hidden_sizes=[64, 64],
                      critic_hidden_sizes=[64, 64],
                      rngs=nnx.Rngs(SEED),
                      action_sampler=NormalSampler(entropy_weight=0.0))
config = ppo.default_config()
config.discounting_factor = 0.0
config.n_envs = 3
training_state = ppo.new_training_state(env, nets, n_envs=config.n_envs, seed=SEED)
#eval_rollout_jit = nnx.jit(rollout.eval_rollout, static_argnums=(0, 2, 3))
for iter in tqdm.trange(500):
    training_state, _ = ppo.ppo_step(env, training_state, 
                                     config.n_envs, config.rollout_length,
                                     config.gae_lambda, config.discounting_factor,
                                     config.clip_range)