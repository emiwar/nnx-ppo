import mujoco_playground
from flax import nnx


from nnx_ppo.networks.modules import PPOActorCritic, MLP, Sequential
from nnx_ppo.networks.future_modules import LSTM, VariationalBottleneck
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
import nnx_ppo.algorithms.ppo

env = mujoco_playground.registry.load("MyFavoriteEnv")
obs_size = env.observation_size
action_size = env.action_size
latent_size = 16
hidden_size = 256

rngs = nnx.Rngs()
net = PPOActorCritic(
    actor=Sequential([
        MLP([obs_size, hidden_size, latent_size], rngs),
        VariationalBottleneck(latent_size, rngs, kl_weight=0.01,),
        LSTM([latent_size, hidden_size], rngs),
        MLP([hidden_size, action_size], rngs),
    ]),
    critic=MLP([obs_size, hidden_size, action_size], rngs),
    action_sampler=NormalTanhSampler(rngs)
)

nnx_ppo.algorithms.ppo.train_ppo(env, net)
