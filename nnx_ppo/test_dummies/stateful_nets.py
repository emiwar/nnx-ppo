from typing import Tuple, Any, Dict
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks import types

class Count(nnx.Variable):
    pass

class RepeatAndCountNet(types.PPONetwork, nnx.Module):
    '''Dummy stateful network that outputs its input as action, and counts
       how many times it has been called.'''
    def __init__(self):
        self.n_calls = Count(0)
    
    def __call__(self, state, obs) -> Tuple[Any, types.PPONetworkOutput]:
        batch_size = obs.shape[0]
        self.n_calls += batch_size
        return (), types.PPONetworkOutput(
            actions=obs,
            loglikelihoods=jp.ones(batch_size),
            regularization_loss=jp.array(0.0),
            value_estimates=jp.ones(batch_size),
            metrics={}
        )
    
    def initialize_state(self, batch_size: int) -> Tuple:
        return ()