"""Recurrent network modules for PPO."""

from typing import Tuple, Optional, Callable

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput

# LSTM carry type: tuple of (hidden_state, cell_state)
LSTMCarry = Tuple[jax.Array, jax.Array]


class LSTM(StatefulModule):
    """LSTM layer that conforms to the StatefulModule interface.

    Wraps flax.nnx.LSTMCell to provide proper state management for RL rollouts.
    The hidden state is reset when the environment resets.

    Example usage:
        lstm = LSTM(in_features=64, hidden_features=128, rngs=nnx.Rngs(0))
        state = lstm.initialize_state(batch_size=32)
        output = lstm(state, x)  # x has shape (32, 64)
        # output.output has shape (32, 128)
        # output.next_state is (h, c) tuple for next timestep
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        rngs: nnx.Rngs,
        *,
        gate_fn: Callable = nnx.sigmoid,
        activation_fn: Callable = nnx.tanh,
        kernel_init: Optional[Callable] = None,
        recurrent_kernel_init: Optional[Callable] = None,
        bias_init: Optional[Callable] = None,
        use_optimized: bool = True,
        trainable_initial_state: bool = False,
    ):
        """Initialize the LSTM layer.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden units (output size).
            rngs: NNX random number generators.
            gate_fn: Activation function for gates (default: sigmoid).
            activation_fn: Activation function for cell state (default: tanh).
            kernel_init: Initializer for input-to-hidden weights.
            recurrent_kernel_init: Initializer for hidden-to-hidden weights.
            bias_init: Initializer for biases.
            use_optimized: If True, use OptimizedLSTMCell which is faster for
                          hidden_features <= 2048.
            trainable_initial_state: If True, the initial hidden and cell states
                          are learnable parameters. Otherwise, they are zeros.
        """
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.trainable_initial_state = trainable_initial_state

        # Build kwargs for cell constructor
        cell_kwargs = {
            "in_features": in_features,
            "hidden_features": hidden_features,
            "rngs": rngs,
            "gate_fn": gate_fn,
            "activation_fn": activation_fn,
        }
        if kernel_init is not None:
            cell_kwargs["kernel_init"] = kernel_init
        if recurrent_kernel_init is not None:
            cell_kwargs["recurrent_kernel_init"] = recurrent_kernel_init
        if bias_init is not None:
            cell_kwargs["bias_init"] = bias_init

        if use_optimized:
            self.cell = nnx.OptimizedLSTMCell(**cell_kwargs)
        else:
            self.cell = nnx.LSTMCell(**cell_kwargs)

        # Trainable initial state (single vector, broadcast to batch)
        if trainable_initial_state:
            self.initial_h = nnx.Param(jp.zeros((hidden_features,)))
            self.initial_c = nnx.Param(jp.zeros((hidden_features,)))

    def __call__(self, state: LSTMCarry, x: jax.Array) -> StatefulModuleOutput:
        """Process input through the LSTM.

        Args:
            state: LSTM carry tuple (hidden_state, cell_state), each with
                   shape (batch_size, hidden_features).
            x: Input array with shape (batch_size, in_features).

        Returns:
            StatefulModuleOutput with:
                - next_state: Updated (h, c) carry tuple
                - output: LSTM output with shape (batch_size, hidden_features)
                - regularization_loss: Zero (LSTM has no regularization)
                - metrics: Empty dict
        """
        next_carry, output = self.cell(state, x)

        return StatefulModuleOutput(
            next_state=next_carry,
            output=output,
            regularization_loss=jp.zeros(x.shape[0]),
            metrics={},
        )

    def initialize_state(self, batch_size: int) -> LSTMCarry:
        """Initialize the LSTM hidden state.

        Args:
            batch_size: Number of parallel environments/sequences.

        Returns:
            Tuple of (hidden_state, cell_state), each with shape
            (batch_size, hidden_features). If trainable_initial_state=True,
            these are learned parameters; otherwise zeros.
        """
        if self.trainable_initial_state:
            h = jp.broadcast_to(self.initial_h[...], (batch_size, self.hidden_features))
            c = jp.broadcast_to(self.initial_c[...], (batch_size, self.hidden_features))
        else:
            h = jp.zeros((batch_size, self.hidden_features))
            c = jp.zeros((batch_size, self.hidden_features))
        return (h, c)

    def reset_state(self, prev_state: LSTMCarry) -> LSTMCarry:
        """Reset LSTM state (called when environment resets).

        Args:
            prev_state: Previous carry state (used to preserve shape).

        Returns:
            Initial carry with same shape as prev_state.
            If trainable_initial_state=True, returns learned initial state
            broadcast to match prev_state shape; otherwise returns zeros.
        """
        if self.trainable_initial_state:
            # Broadcast learned initial state to match prev_state shape
            h = jp.broadcast_to(self.initial_h[...], prev_state[0].shape)
            c = jp.broadcast_to(self.initial_c[...], prev_state[1].shape)
            return (h, c)
        else:
            # Return zeros with same shape as prev_state
            return (jp.zeros_like(prev_state[0]), jp.zeros_like(prev_state[1]))
