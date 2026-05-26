from typing import Any
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks import types


class Count(nnx.Variable):
    pass


class RepeatAndCountNet(types.StatefulModule):
    """Dummy stateful network that outputs its input as action, and counts
    how many times it has been called."""

    def __init__(self):
        self.n_calls = Count(0)

    def __call__(
        self,
        state,
        obs,
        rollout_extras: Any = None,
    ) -> types.StatefulModuleOutput:
        batch_size = obs.shape[0]
        self.n_calls[...] += batch_size
        return types.StatefulModuleOutput(
            next_state=(),
            output=types.PPONetworkOutput(
                actions=obs,
                loglikelihoods=jp.ones(batch_size),
                value_estimates=jp.ones(batch_size),
            ),
            regularization_loss=jp.array(0.0),
            metrics={},
            rollout_extras=None,
        )

    def initialize_state(self, batch_size: int) -> tuple:
        return ()
