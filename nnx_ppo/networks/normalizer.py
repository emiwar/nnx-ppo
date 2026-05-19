from typing import Any
from warnings import deprecated

import jax
import jax.numpy as jp
from flax import nnx
from jaxtyping import ScalarLike

from nnx_ppo.algorithms.types import Transition
from nnx_ppo.networks.types import (
    Context,
    StatefulModule,
    StatefulModuleOutput,
)


class NormalizerStatistics(nnx.Variable):
    pass


class Normalizer(StatefulModule):
    """Online (Welford) input normalizer.

    Forward pass standardises ``x`` using the running ``mean`` and standard
    deviation derived from ``M2`` and ``counter``. The running statistics
    are updated *only* when ``__call__`` is invoked with
    ``context=Context.STATS_UPDATE``; in all other contexts the live params
    are read-only.

    The ``STATS_UPDATE`` pass is invoked by ``ppo_step`` after the gradient
    phase: the rollout is replayed through the network once more (using
    stored ``raw_action`` for samplers, just like ``LOSS_REPLAY``), and on
    that pass each Normalizer sees the exact same activations it saw during
    rollout and incrementally accumulates them into ``mean``/``M2``/
    ``counter``. The forward output of this pass is discarded.

    Placing a Normalizer anywhere in the network — behind a ``Delay``,
    inside a population of a graph, after an encoder — works automatically:
    the input ``x`` it receives during STATS_UPDATE is, by construction,
    the same as during rollout.
    """

    def __init__(self, shape):
        if isinstance(shape, (tuple, list, int)):
            self.mean = NormalizerStatistics(jp.zeros(shape))
            self.M2 = NormalizerStatistics(jp.zeros(shape))
        else:
            self.mean = NormalizerStatistics(jax.tree.map(jp.zeros, shape))
            self.M2 = NormalizerStatistics(jax.tree.map(jp.zeros, shape))
        self.counter = NormalizerStatistics(jp.array(0.0))
        self.epsilon = 1e-6

    def __call__(
        self,
        state: tuple[()],
        x: Any,
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        if context == Context.STATS_UPDATE:
            self._welford_step(x)
        std = jax.lax.cond(
            self.counter.get_value() > 0,
            self._M2_to_std,
            lambda M2: jax.tree.map(lambda x: jp.full(x.shape, 10.0), M2),
            self.M2.get_value(),
        )
        output = jax.tree.map(
            lambda x, m, s: (x - m) / s, x, self.mean.get_value(), std
        )
        return StatefulModuleOutput(
            next_state=(), output=output, regularization_loss=jp.array(0.0), metrics={}
        )

    def _M2_to_std(self, M2):
        return jax.tree.map(
            lambda x: jp.sqrt(jp.maximum(x / self.counter.get_value(), self.epsilon)),
            M2,
        )

    def _welford_step(self, x: Any) -> None:
        """Per-step batched Welford update.

        ``x`` is a single timestep slice with shape ``[B, *feat]`` per leaf.
        Updates ``self.mean``, ``self.M2``, ``self.counter`` in place. The
        update is mathematically associative across steps, so applying it
        once per timestep over a rollout produces the same end state as
        applying it once on the whole rollout.
        """
        leaves = jax.tree.leaves(x)
        if not leaves:
            return
        # Leaves are [B, *feat] in a single-step call.
        batch_count = leaves[0].shape[0]
        new_count = self.counter.get_value() + batch_count
        frac = batch_count / new_count

        batch_mean = jax.tree.map(lambda v: jp.mean(v, axis=0), x)
        old_mean = self.mean.get_value()
        delta_old = jax.tree.map(lambda bm, m: bm - m, batch_mean, old_mean)
        new_mean = jax.tree.map(lambda m, d: m + d * frac, old_mean, delta_old)
        self.mean.set_value(new_mean)

        delta_new = jax.tree.map(lambda bm, m: bm - m, batch_mean, new_mean)
        batch_var = jax.tree.map(lambda v: jp.var(v, axis=0), x)
        old_M2 = self.M2.get_value()
        new_M2 = jax.tree.map(
            lambda m2, bv, d_old, d_new: m2
            + batch_count * bv
            + batch_count * d_old * d_new,
            old_M2,
            batch_var,
            delta_old,
            delta_new,
        )
        self.M2.set_value(new_M2)
        self.counter.set_value(new_count)

    @deprecated(
        "Internal helper for the legacy update_statistics(rollout, total_steps) "
        "shim. Will be removed together with that shim once experiments are "
        "migrated to the context=Context.STATS_UPDATE forward-pass path."
    )
    def _welford_full_rollout(self, obs: Any) -> None:
        """Backwards-compat helper. Treats ``obs`` as a ``[T, B, *feat]``
        rollout (or pytree thereof) and folds it into the live statistics
        in one shot using the per-rollout Welford update formula. Equivalent
        to applying ``_welford_step`` T times, but cheaper.
        """
        leaves = jax.tree.leaves(obs)
        if not leaves:
            return
        batch_count = leaves[0].shape[0] * leaves[0].shape[1]
        new_count = self.counter.get_value() + batch_count
        frac = batch_count / new_count

        batch_mean = jax.tree.map(lambda v: jp.mean(v, axis=(0, 1)), obs)
        old_mean = self.mean.get_value()
        delta_old = jax.tree.map(lambda bm, m: bm - m, batch_mean, old_mean)
        new_mean = jax.tree.map(lambda m, d: m + d * frac, old_mean, delta_old)
        self.mean.set_value(new_mean)

        delta_new = jax.tree.map(lambda bm, m: bm - m, batch_mean, new_mean)
        batch_var = jax.tree.map(lambda v: jp.var(v, axis=(0, 1)), obs)
        old_M2 = self.M2.get_value()
        new_M2 = jax.tree.map(
            lambda m2, bv, d_old, d_new: m2
            + batch_count * bv
            + batch_count * d_old * d_new,
            old_M2,
            batch_var,
            delta_old,
            delta_new,
        )
        self.M2.set_value(new_M2)
        self.counter.set_value(new_count)

    @deprecated(
        "Use the context=Context.STATS_UPDATE forward pass instead. This "
        "shim is retained only for callers in vnl-experiments that haven't "
        "migrated to the context-based API and will be removed once they "
        "have."
    )
    def update_statistics(
        self, last_rollout: Transition, total_steps: ScalarLike
    ) -> None:
        """Backwards-compat entry point used by older callers (e.g. NerveNet
        variants that haven't migrated to the context-based API). Treats
        ``last_rollout.obs`` as the rollout this normalizer saw and folds
        it into the live statistics.

        New code should pass ``context=Context.STATS_UPDATE`` to the
        network's forward pass instead, which lets the walker invoke
        ``__call__`` per timestep with the correctly-routed input."""
        self._welford_full_rollout(last_rollout.obs)
