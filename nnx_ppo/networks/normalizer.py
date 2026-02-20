import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.algorithms.types import Transition
from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput


class NormalizerStatistics(nnx.Variable):
    pass


class Normalizer(StatefulModule):

    def __init__(self, shape):
        if isinstance(shape, (tuple, list, int)):
            self.mean = NormalizerStatistics(jp.zeros(shape))
            self.M2 = NormalizerStatistics(jp.zeros(shape))
        else:
            self.mean = NormalizerStatistics(jax.tree.map(jp.zeros_like, shape))
            self.M2 = NormalizerStatistics(jax.tree.map(jp.zeros_like, shape))
        self.counter = NormalizerStatistics(jp.array(0.0))
        self.epsilon = 1e-6

    def __call__(self, state, x):
        # Compute variance from M2
        std = jax.lax.cond(
            self.counter[...] > 0,
            self.M2_to_std,
            lambda M2: jax.tree.map(lambda x: jp.full(x.shape, 10.0), M2),
            self.M2[...],
        )
        output = jax.tree.map(lambda x, m, s: (x - m) / s, x, self.mean[...], std)
        return StatefulModuleOutput(
            next_state=(), output=output, regularization_loss=jp.array(0.0), metrics={}
        )

    def M2_to_std(self, M2):
        return jax.tree.map(
            lambda x: jp.sqrt(jp.maximum(x / self.counter, self.epsilon)), M2
        )

    def update_statistics(
        self, last_rollout: Transition, total_steps: jax.Array
    ) -> None:
        obs = last_rollout.obs
        batch_count = last_rollout.done.size
        new_count = self.counter[...] + batch_count
        frac = batch_count / new_count

        # Welford's algorithm for batched updates
        batch_mean = jax.tree.map(lambda x: jp.mean(x, axis=(0, 1)), obs)
        delta_old = jax.tree.map(
            lambda batch_mean, old: batch_mean - old, batch_mean, self.mean[...]
        )
        self.mean[...] = jax.tree.map(
            lambda old, delta: old + delta * frac, self.mean[...], delta_old
        )
        delta_new = jax.tree.map(
            lambda batch_mean, old: batch_mean - old, batch_mean, self.mean[...]
        )

        batch_var = jax.tree.map(lambda x: jp.var(x, axis=(0, 1)), obs)
        self.M2[...] = jax.tree.map(
            lambda old, batch_var, delta_old, delta_new: old
            + batch_count * batch_var
            + batch_count * delta_old * delta_new,
            self.M2[...],
            batch_var,
            delta_old,
            delta_new,
        )

        # Update counter
        self.counter[...] += batch_count
