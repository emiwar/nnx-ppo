"""Recurrent network modules for PPO.

Each class here wraps one of flax's ``nnx`` RNN cells as a
:class:`~nnx_ppo.networks.types.StatefulModule`, so the cell's carry becomes the
module's explicit carry state and is threaded, sliced and reset by the PPO
machinery like any other carry:

* :class:`LSTM`      — long short-term memory, two-slot carry.
* :class:`GRU`       — gated recurrent unit, one-slot carry.
* :class:`SimpleRNN` — vanilla (Elman) RNN, one-slot carry.

All three share :class:`RecurrentCell`, which implements the whole
``StatefulModule`` carry contract once. To wrap a cell that is not covered here,
subclass :class:`RecurrentCell` — see its docstring.
"""
from collections.abc import Callable
from typing import Any, Optional

import jax.numpy as jp
from flax import nnx
from jaxtyping import Array, Float

from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput

#: Carry of a single-slot cell (:class:`GRU`, :class:`SimpleRNN`): one array.
RecurrentCarry = Float[Array, "*batch hidden"]

#: Carry of :class:`LSTM`. Two arrays; see the note on ordering in ``LSTM``.
LSTMCarry = tuple[Float[Array, "*batch hidden"], Float[Array, "*batch hidden"]]


def _cell_kwargs(
    in_features: int,
    hidden_features: int,
    rngs: nnx.Rngs,
    **optional: Any,
) -> dict[str, Any]:
    """Cell constructor kwargs, dropping any ``None`` so flax's own defaults win.

    The initialiser defaults differ per argument and per cell (e.g. ``kernel_init``
    is a ``variance_scaling`` while ``recurrent_kernel_init`` is ``orthogonal``),
    so passing ``None`` through explicitly would override them with the wrong
    thing. Omitting the key is the only way to say "use the cell's default".
    """
    kwargs: dict[str, Any] = {
        "in_features": in_features,
        "hidden_features": hidden_features,
        "rngs": rngs,
    }
    kwargs.update({k: v for k, v in optional.items() if v is not None})
    return kwargs


class RecurrentCell(StatefulModule):
    """``StatefulModule`` wrapper around a flax ``nnx`` RNN cell.

    Subclasses build the cell, hand it to ``super().__init__``, and declare
    :attr:`carry_names` — one name per slot of the cell's carry, **in the order
    the cell itself expects them**. That single declaration determines everything:

    * the carry is a bare array when there is one name, a tuple when there are
      more, which is exactly what each flax cell accepts and returns;
    * with ``trainable_initial_state=True`` each slot gets a learned parameter
      attribute called ``initial_<name>``.

    A minimal subclass is therefore::

        class GRU(RecurrentCell):
            carry_names = ("h",)

            def __init__(self, in_features, hidden_features, rngs, **kw):
                cell = nnx.GRUCell(in_features=in_features,
                                   hidden_features=hidden_features, rngs=rngs)
                super().__init__(cell, in_features, hidden_features, **kw)

    ``reset_state`` is written purely with ``jp.zeros_like`` / ``jp.broadcast_to``
    and so preserves the shape it is given. That matters because PPO calls it two
    different ways: on the whole batch during rollout (``rollout.unroll_env``
    selects per env with ``tree_where``) and under ``jax.vmap`` on a single-env
    slice during loss replay (``ppo.ppo_loss``). Both must work.

    Args:
        cell: The flax ``nnx`` RNN cell to wrap. Must be callable as
            ``cell(carry, x) -> (next_carry, output)``.
        in_features: Number of input features.
        hidden_features: Number of hidden units, and hence the output size.
        trainable_initial_state: If True, the initial carry is a learned
            parameter per slot rather than zeros. Because ``reset_state`` is
            called inside the loss-replay scan, gradients do reach these.
    """

    #: One name per slot of the wrapped cell's carry, in the cell's own order.
    carry_names: tuple[str, ...] = ("h",)

    def __init__(
        self,
        cell: Any,
        in_features: int,
        hidden_features: int,
        *,
        trainable_initial_state: bool = False,
    ):
        self.cell = cell
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.trainable_initial_state = trainable_initial_state

        if trainable_initial_state:
            for name in self.carry_names:
                setattr(self, f"initial_{name}", nnx.Param(jp.zeros((hidden_features,))))

    # -- carry plumbing ----------------------------------------------------
    # A one-slot carry is the bare array, not a length-1 tuple, because that is
    # what flax's single-slot cells return and expect.

    def _pack(self, slots: list[Array]) -> Any:
        return slots[0] if len(self.carry_names) == 1 else tuple(slots)

    def _unpack(self, carry: Any) -> list[Array]:
        return [carry] if len(self.carry_names) == 1 else list(carry)

    def _learned_initial(self) -> list[Array]:
        return [getattr(self, f"initial_{name}")[...] for name in self.carry_names]

    # -- StatefulModule ----------------------------------------------------

    def __call__(
        self,
        state: Any,
        x: Float[Array, "batch {self.in_features}"],
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        """Advance the cell one step.

        Args:
            state: The carry, shaped as described in :attr:`carry_names`; every
                slot has shape ``(batch, hidden_features)``.
            x: Input, shape ``(batch, in_features)``.
            rollout_extras: Unused — a recurrent cell is deterministic given its
                carry, so it needs no replay channel.

        Returns:
            ``StatefulModuleOutput`` whose ``output`` is the cell's output of
            shape ``(batch, hidden_features)`` and whose ``next_state`` is the
            updated carry.
        """
        next_carry, output = self.cell(state, x)

        return StatefulModuleOutput(
            next_state=next_carry,
            output=output,
            regularization_loss=jp.zeros(x.shape[0]),
            metrics={},
            rollout_extras=None,
        )

    def initialize_state(self, batch_size: int) -> Any:
        """Build a fresh carry for ``batch_size`` parallel environments.

        Returns zeros, or the learned initial state broadcast over the batch when
        ``trainable_initial_state=True``.
        """
        shape = (batch_size, self.hidden_features)
        if self.trainable_initial_state:
            return self._pack(
                [jp.broadcast_to(v, shape) for v in self._learned_initial()]
            )
        return self._pack([jp.zeros(shape) for _ in self.carry_names])

    def reset_state(self, prev_state: Any) -> Any:
        """Reset the carry at an episode boundary, preserving ``prev_state``'s shape.

        Shape is taken from ``prev_state`` rather than from ``batch_size`` so this
        works both on the full batch and under ``vmap`` on a single-env slice.
        """
        slots = self._unpack(prev_state)
        if self.trainable_initial_state:
            return self._pack(
                [
                    jp.broadcast_to(v, slot.shape)
                    for v, slot in zip(self._learned_initial(), slots)
                ]
            )
        return self._pack([jp.zeros_like(slot) for slot in slots])


class LSTM(RecurrentCell):
    """LSTM layer conforming to the ``StatefulModule`` interface.

    Wraps ``nnx.OptimizedLSTMCell`` (or ``nnx.LSTMCell``) so its carry is managed
    across RL rollouts and reset when the environment resets.

    .. note::
        The carry is a two-tuple in **flax's** order, which is ``(c, h)`` — cell
        state first, hidden state second (see ``nnx.LSTMCell.__call__``). The
        learned initial-state parameters are historically named ``initial_h`` and
        ``initial_c`` and occupy slots 0 and 1 respectively, so ``initial_h`` in
        fact seeds the *cell* state and ``initial_c`` the *hidden* state. Both are
        zero-initialised and symmetric in the API, so this is a naming wart rather
        than a behavioural bug; the names are kept because renaming them would
        invalidate existing checkpoints.

    Example::

        lstm = LSTM(in_features=64, hidden_features=128, rngs=nnx.Rngs(0))
        state = lstm.initialize_state(batch_size=32)
        out = lstm(state, x)          # x: (32, 64)
        out.output                    # (32, 128)
        out.next_state                # carry for the next timestep

    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden units (output size).
        rngs: NNX random number generators.
        gate_fn: Activation for the gates (default sigmoid).
        activation_fn: Activation for the cell state (default tanh).
        kernel_init: Initialiser for input-to-hidden weights. ``None`` keeps
            flax's default.
        recurrent_kernel_init: Initialiser for hidden-to-hidden weights.
            ``None`` keeps flax's default.
        bias_init: Initialiser for biases. ``None`` keeps flax's default.
        use_optimized: If True use ``nnx.OptimizedLSTMCell``, which is faster for
            ``hidden_features <= 2048``.
        trainable_initial_state: If True the initial carry is learned rather than
            zeros.
    """

    carry_names = ("h", "c")

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
        cell_cls = nnx.OptimizedLSTMCell if use_optimized else nnx.LSTMCell
        cell = cell_cls(
            **_cell_kwargs(
                in_features,
                hidden_features,
                rngs,
                gate_fn=gate_fn,
                activation_fn=activation_fn,
                kernel_init=kernel_init,
                recurrent_kernel_init=recurrent_kernel_init,
                bias_init=bias_init,
            )
        )
        super().__init__(
            cell,
            in_features,
            hidden_features,
            trainable_initial_state=trainable_initial_state,
        )


class GRU(RecurrentCell):
    """Gated recurrent unit conforming to the ``StatefulModule`` interface.

    Wraps ``nnx.GRUCell``. The carry is a single ``(batch, hidden_features)``
    array — the GRU merges the LSTM's cell and hidden state into one — so it costs
    roughly three quarters of an LSTM's parameters and half its carry.

    Example::

        gru = GRU(in_features=64, hidden_features=128, rngs=nnx.Rngs(0))
        state = gru.initialize_state(batch_size=32)
        out = gru(state, x)           # x: (32, 64)
        out.output                    # (32, 128)

    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden units (output size).
        rngs: NNX random number generators.
        gate_fn: Activation for the update/reset gates (default sigmoid).
        activation_fn: Activation for the candidate state (default tanh).
        kernel_init: Initialiser for input-to-hidden weights. ``None`` keeps
            flax's default.
        recurrent_kernel_init: Initialiser for hidden-to-hidden weights.
            ``None`` keeps flax's default.
        bias_init: Initialiser for biases. ``None`` keeps flax's default.
        trainable_initial_state: If True the initial carry is learned rather than
            zeros, exposed as ``initial_h``.
    """

    carry_names = ("h",)

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
        trainable_initial_state: bool = False,
    ):
        cell = nnx.GRUCell(
            **_cell_kwargs(
                in_features,
                hidden_features,
                rngs,
                gate_fn=gate_fn,
                activation_fn=activation_fn,
                kernel_init=kernel_init,
                recurrent_kernel_init=recurrent_kernel_init,
                bias_init=bias_init,
            )
        )
        super().__init__(
            cell,
            in_features,
            hidden_features,
            trainable_initial_state=trainable_initial_state,
        )


class SimpleRNN(RecurrentCell):
    """Vanilla (Elman) RNN conforming to the ``StatefulModule`` interface.

    Wraps ``nnx.SimpleCell``: ``h' = activation_fn(W x + U h + b)``, with a single
    ``(batch, hidden_features)`` carry and no gating. Useful mainly as a baseline
    against :class:`GRU` / :class:`LSTM` — without gates it is the one to reach
    for when the question is how much of a result the gating is responsible for,
    not when you want the best-performing policy.

    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden units (output size).
        rngs: NNX random number generators.
        activation_fn: Elementwise nonlinearity (default tanh). Note there is no
            ``gate_fn`` — an Elman cell has no gates.
        residual: If True the cell adds its input projection to the carry rather
            than replacing it, which lengthens the gradient path through time.
        kernel_init: Initialiser for input-to-hidden weights. ``None`` keeps
            flax's default.
        recurrent_kernel_init: Initialiser for hidden-to-hidden weights.
            ``None`` keeps flax's default (orthogonal, which matters more here
            than for the gated cells).
        bias_init: Initialiser for biases. ``None`` keeps flax's default.
        trainable_initial_state: If True the initial carry is learned rather than
            zeros, exposed as ``initial_h``.
    """

    carry_names = ("h",)

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        rngs: nnx.Rngs,
        *,
        activation_fn: Callable = nnx.tanh,
        residual: bool = False,
        kernel_init: Optional[Callable] = None,
        recurrent_kernel_init: Optional[Callable] = None,
        bias_init: Optional[Callable] = None,
        trainable_initial_state: bool = False,
    ):
        cell = nnx.SimpleCell(
            **_cell_kwargs(
                in_features,
                hidden_features,
                rngs,
                activation_fn=activation_fn,
                residual=residual,
                kernel_init=kernel_init,
                recurrent_kernel_init=recurrent_kernel_init,
                bias_init=bias_init,
            )
        )
        super().__init__(
            cell,
            in_features,
            hidden_features,
            trainable_initial_state=trainable_initial_state,
        )
