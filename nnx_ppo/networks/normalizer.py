from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import (
    StatefulModule,
    StatefulModuleOutput,
)


class NormalizerStatistics(nnx.Variable):
    pass


def _canonicalize(obj: Any) -> Any:
    """Recursively convert any Mapping (e.g. OrderedDict, FrozenDict,
    ConfigDict) to a plain ``dict``. JAX registers plain ``dict`` and
    ``OrderedDict`` as distinct pytree node types, so a Normalizer
    initialised from one and called with the other fails to ``tree.map``.
    Canonicalising both sides to plain ``dict`` removes the mismatch.
    Lists/tuples and other types are preserved.
    """
    if isinstance(obj, Mapping):
        return {k: _canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_canonicalize(v) for v in obj)
    return obj


class Normalizer(StatefulModule):
    """Online (Welford) input normalizer.

    Forward pass standardises ``x`` using the running ``mean`` and standard
    deviation derived from ``M2`` and ``counter``. The running statistics
    are read-only in ``__call__`` — never written from the forward path.

    Stats are updated once per training step via :meth:`update_statistics`,
    which receives the rollout's history of normalizer inputs (one
    ``[T, B, *feat]`` slice per leaf) and folds it in with a single batched
    Welford merge. The values it sees are exactly the activations the
    Normalizer received during ROLLOUT (the forward pass emits them as
    ``rollout_extras``), so placing the Normalizer anywhere — behind a
    ``Delay``, inside a graph population, after an encoder — works
    automatically.
    """

    def __init__(self, shape):
        if isinstance(shape, (tuple, list, int)):
            self.mean = NormalizerStatistics(jp.zeros(shape))
            self.M2 = NormalizerStatistics(jp.zeros(shape))
        else:
            shape = _canonicalize(shape)
            self.mean = NormalizerStatistics(jax.tree.map(jp.zeros, shape))
            self.M2 = NormalizerStatistics(jax.tree.map(jp.zeros, shape))
        self.counter = NormalizerStatistics(jp.array(0.0))
        self.epsilon = 1e-6

    def __call__(
        self,
        state: tuple[()],
        x: Any,
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        # Canonicalise any incoming Mapping (OrderedDict, FrozenDict, ...)
        # to plain dict so jax.tree.map aligns with self.mean / self.M2.
        x = _canonicalize(x)
        std = jax.lax.cond(
            self.counter.get_value() > 0,
            self._M2_to_std,
            lambda M2: jax.tree.map(lambda x: jp.full(x.shape, 10.0), M2),
            self.M2.get_value(),
        )
        output = jax.tree.map(
            lambda x, m, s: (x - m) / s, x, self.mean.get_value(), std
        )
        # Always emit the normalised input as rollout_extras; update_statistics
        # will fold the [T, B, ...] history into running stats after the loss
        # step. The eval/inference path discards the emission.
        return StatefulModuleOutput(
            next_state=(),
            output=output,
            regularization_loss=jp.array(0.0),
            metrics={},
            rollout_extras=x,
        )

    def _M2_to_std(self, M2):
        return jax.tree.map(
            lambda x: jp.sqrt(jp.maximum(x / self.counter.get_value(), self.epsilon)),
            M2,
        )

    def update_statistics(self, rollout_extras: Any) -> None:
        """Fold the rollout's worth of normalizer inputs into running stats.

        ``rollout_extras`` is a pytree matching the structure the Normalizer
        emitted on each step, with an additional leading time dimension —
        leaves have shape ``[T, B, *feat]``. We flatten T*B and apply one
        batched Welford merge.
        """
        leaves = jax.tree.leaves(rollout_extras)
        # Flatten time and batch axes into a single sample axis [N, *feat].
        flat = jax.tree.map(
            lambda v: v.reshape((-1,) + v.shape[2:]), rollout_extras
        )
        n = leaves[0].shape[0] * leaves[0].shape[1]
        new_count = self.counter.get_value() + n
        frac = n / new_count

        batch_mean = jax.tree.map(lambda v: jp.mean(v, axis=0), flat)
        batch_M2 = jax.tree.map(
            lambda v, bm: jp.sum(jp.square(v - bm), axis=0), flat, batch_mean
        )

        old_mean = self.mean.get_value()
        delta = jax.tree.map(lambda bm, m: bm - m, batch_mean, old_mean)
        new_mean = jax.tree.map(lambda m, d: m + d * frac, old_mean, delta)

        old_M2 = self.M2.get_value()
        # Standard batched Welford merge: M2_combined = M2_a + M2_b + d^2 * n_a * n_b / (n_a + n_b)
        new_M2 = jax.tree.map(
            lambda m2, bm2, d: m2
            + bm2
            + (d * d) * self.counter.get_value() * n / new_count,
            old_M2,
            batch_M2,
            delta,
        )
        self.mean.set_value(new_mean)
        self.M2.set_value(new_M2)
        self.counter.set_value(new_count)
