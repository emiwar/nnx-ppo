
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from flax import nnx

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

# Do a rollout to initialize normalizers etc before starting actual training.
net_state, env_state, rollout_data = rollout.unroll_env(train_env,
                                                        training_state.env_states,
                                                        training_state.networks,
                                                        training_state.network_states,
                                                        config.rollout_length,
                                                        training_state.rng_key)
