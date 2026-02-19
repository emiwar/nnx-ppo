"""Feedforward network layers."""
from typing import Tuple, Callable, Optional

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput


class Dense(StatefulModule):
    """Single dense layer with optional activation, implementing StatefulModule.

    This is a thin wrapper around nnx.Linear that conforms to the StatefulModule
    interface, allowing it to be used in Sequential containers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        activation: Optional[Callable] = None,
        **linear_kwargs
    ):
        """Initialize the dense layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            rngs: NNX random number generators.
            activation: Optional activation function applied after the linear transform.
            **linear_kwargs: Additional arguments passed to nnx.Linear (e.g., kernel_init).
        """
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs, **linear_kwargs)
        self.activation = activation

    def __call__(self, state: Tuple, x: jax.Array) -> StatefulModuleOutput:
        y = self.linear(x)
        if self.activation is not None:
            y = self.activation(y)
        return StatefulModuleOutput(state, y, jp.array(0.0), {})
