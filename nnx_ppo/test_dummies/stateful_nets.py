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
        self.n_calls += 1
        return (), types.PPONetworkOutput(
            actions=obs,
            loglikelihoods=jp.array(1.0),
            regularization_loss=jp.array(0.0),
            value_estimates=jp.array(1.0),
            metrics={}
        )
    
    def initialize_state(self, rng: jax.Array) -> Tuple:
        return ()
    
    def reset_state(self, network_state) -> Tuple:
        return ()
