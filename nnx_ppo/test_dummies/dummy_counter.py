from typing import Tuple, Any, Dict
import jax
import jax.numpy as jp
import mujoco_playground
from flax import nnx

from nnx_ppo.networks import types

class DummyCounterEnv:
    '''Dummy environment that gives reward 1.0 if the action is the number
       of steps since the last reset, and 0.0 otherwise. Observation is always
       [0.0].'''

    def reset(self, rng):
        return mujoco_playground.State(
            data = {"current_step": jp.array(0),
                    "reset_step": jax.random.randint(rng, (), 3, 10)},
            obs = jp.zeros(1),
            info = {'current_step': 0},
            reward = jp.array(1.0),
            done = jp.array(0.0),
            metrics = {}
        )
    
    def step(self, state: mujoco_playground.State, action: jax.Array):
        data = {"current_step": state.data["current_step"] + 1,
                "reset_step": state.data["reset_step"]}
        done = jp.astype(data["current_step"] >= data["reset_step"], float)
        return mujoco_playground.State(
            data = data,
            obs = jp.zeros(1),
            info={'current_step': data["current_step"],},
            reward=jp.where(action==data["current_step"], 1.0, 0.0),
            done=done,
            metrics=state.metrics
        )
    
class DummyCounterNet(types.PPONetwork, nnx.Module):
    '''Dummy stateful network that always outputs the number of steps since its
       last reset, independent of its input.'''
    
    def __call__(self, state, obs) -> Tuple[Any, types.PPONetworkOutput]:
        old_counter = state["counter_state"]["counter"]
        new_counter = old_counter + 1
        new_state = {
            "counter_state": {
                "counter": new_counter
            }
        }
        return new_state, types.PPONetworkOutput(
            actions=new_counter,
            raw_actions=new_counter,
            loglikelihoods=jp.ones_like(old_counter, dtype=float),
            regularization_loss=jp.array(0.0),
            value_estimates=jp.ones_like(old_counter, dtype=float),
            metrics={}
        )
    
    def initialize_state(self, batch_size: int) -> Dict:
        return {"counter_state": {"counter": jp.zeros(batch_size, dtype=int)}}
