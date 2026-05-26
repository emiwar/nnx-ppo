"""k-step delay layer.

Generalises the per-network ``DelayedObsNetwork`` wrapper (previously in
``vnl-experiments``) into a composable ``StatefulModule`` that can be
inserted anywhere a layer fits: as an element of a ``Sequential``, as the
transform of a graph connection, or as a wrapper around any module
producing an array / pytree of arrays.
"""

from typing import Any

import jax
import jax.numpy as jp

from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput


class Delay(StatefulModule):
    """k-step delay.

    Output at time t is the input from time t - k_steps. Before the buffer
    fills (t < k_steps), output is ``initial_value`` (default zero).

    Carry state is a dict::

        {"buffer": <pytree mirroring input, leaves [B, k_steps, *leaf]>,
         "idx":    <[B] int32 circular write pointer>}

    ``reset_state`` zeros both buffer and idx, which is the correct
    behaviour at episode boundaries.

    Example:
        sample_obs = jax.jit(env.reset)(jax.random.key(0)).obs
        net = Sequential([Delay(sample_obs, k_steps=5), inner_network])
    """

    def __init__(self, sample_input: Any, k_steps: int, initial_value: float = 0.0):
        """Initialise the delay buffer's pytree spec from a sample input.

        Args:
            sample_input: A single *unbatched* example of the input PyTree.
                Used only to capture the leaf shapes, dtypes, and tree
                structure for buffer allocation. The values themselves are
                not retained.
            k_steps: Delay length in steps. Must be >= 1.
            initial_value: Value used to fill the buffer before it has been
                written ``k_steps`` times (and on ``reset_state``).
        """
        if k_steps < 1:
            raise ValueError(f"k_steps must be >= 1, got {k_steps}")
        self.k_steps = k_steps
        self.initial_value = initial_value
        leaves, self._treedef = jax.tree_util.tree_flatten(sample_input)
        self._leaf_shapes = tuple(leaf.shape for leaf in leaves)
        self._leaf_dtypes = tuple(leaf.dtype for leaf in leaves)

    def __call__(
        self,
        state: dict,
        x: Any,
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        idx = state["idx"]
        batch_size = idx.shape[0]
        arange = jp.arange(batch_size)

        delayed = jax.tree.map(lambda b: b[arange, idx], state["buffer"])
        new_buffer = jax.tree.map(
            lambda b, x_: b.at[arange, idx].set(x_), state["buffer"], x
        )
        new_idx = (idx + 1) % self.k_steps

        return StatefulModuleOutput(
            next_state={"buffer": new_buffer, "idx": new_idx},
            output=delayed,
            regularization_loss=jp.zeros(batch_size),
            metrics={},
            rollout_extras=None,
        )

    def initialize_state(self, batch_size: int) -> dict:
        buffer_leaves = [
            jp.full((batch_size, self.k_steps) + shape, self.initial_value, dtype)
            for shape, dtype in zip(self._leaf_shapes, self._leaf_dtypes)
        ]
        buffer = jax.tree_util.tree_unflatten(self._treedef, buffer_leaves)
        return {"buffer": buffer, "idx": jp.zeros(batch_size, jp.int32)}

    def reset_state(self, prev_state: dict) -> dict:
        return {
            "buffer": jax.tree.map(
                lambda b: jp.full_like(b, self.initial_value), prev_state["buffer"]
            ),
            "idx": jp.zeros_like(prev_state["idx"]),
        }
