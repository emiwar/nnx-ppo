from typing import Tuple, Any, Dict, Sequence
import jax
import jax.numpy as jp
import mujoco_playground
from flax import nnx

from nnx_ppo.networks import types

class MoveToCenterEnv:
    '''Dummy environment with continuous steps in 2D. The agent is rewarded for being
       close to the origin. Episode ends if the agent goes too far away.'''

    def __init__(self, reward_falloff: float = 0.5, border_radius=2.0):
        self.reward_falloff = reward_falloff
        self.border_radius = border_radius

    def reset(self, rng):
        pos = jax.random.normal(rng, (2,))
        data = dict(pos=pos)
        return self._get_state(data)
    
    def step(self, state: mujoco_playground.State, action: jax.Array):
        state = state.replace(
            data = dict(pos=state.data["pos"] + action),
        )
        return self._get_state(state.data)
    
    def _get_state(self, data):
        d_sqr = jp.square(data["pos"]).sum()
        reward = jp.exp(-(d_sqr/(self.reward_falloff**2)/2))
        return mujoco_playground.State(
            data = data,
            obs = data["pos"],
            info = {},
            reward = reward,
            done = jp.where(d_sqr > self.border_radius**2, 1.0, 0.0),
            metrics = {}
        )
    
    @property
    def observation_size(self):
        return 2
    
    @property
    def action_size(self):
        return 2