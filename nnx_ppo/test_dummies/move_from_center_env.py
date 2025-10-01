from typing import Tuple, Any, Dict, Sequence
import jax
import jax.numpy as jp
import mujoco_playground
from flax import nnx

from nnx_ppo.networks import types

class MoveFromCenterEnv:
    '''Dummy environment with continuous steps in 2D. The agent is penalized for being
       close to the origin. Episode ends if the agent reaches the border. In this env,
       short lifespans are preferred.'''

    def __init__(self, border_radius=2.0):
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
        d = jp.linalg.norm(data["pos"])
        #Reward is negative, encouraging the agent to "escape" to the border
        reward = d/self.border_radius - 1.0
        return mujoco_playground.State(
            data = data,
            obs = data["pos"],
            info = {},
            reward = reward,
            done = jp.where(d > self.border_radius, 1.0, 0.0),
            metrics = {}
        )
    
    @property
    def observation_size(self):
        return 2
    
    @property
    def action_size(self):
        return 2