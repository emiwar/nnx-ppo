from typing import Tuple, Any, Dict, Sequence
import jax
import jax.numpy as jp
import mujoco_playground
from flax import nnx

from nnx_ppo.networks import types

class ParrotEnv:
    '''Dummy environment that gives maximal reward when action=last_obs'''

    def __init__(self, obs_size: Sequence[int] = (3,), reward_falloff: float = 0.5):
        self.obs_size = obs_size
        self.reward_falloff = reward_falloff

    def reset(self, rng):
        return self._get_obs(rng, None, None)
    
    def step(self, state: mujoco_playground.State, action: jax.Array):
        return self._get_obs(state.data["rng_key"], state.obs, action)
    
    def _get_obs(self, rng_key, obs, action):
        if action is not None:
            d_sqr = jp.square(action - obs).sum()
            reward = jp.exp(-(d_sqr/(self.reward_falloff**2)/2))
        else:
            reward = jp.array(0.0)
        obs_key, new_key = jax.random.split(rng_key)
        next_obs = jp.tanh(jax.random.normal(obs_key, self.obs_size))
        return mujoco_playground.State(
            data = dict(rng_key=new_key),
            obs = next_obs,
            info = {},
            reward = reward,
            done = jp.array(0.0),
            metrics = {}
        )
    
    @property
    def observation_size(self):
        return self.obs_size
    
    @property
    def action_size(self):
        #Action size is the same as obs size
        return self.obs_size