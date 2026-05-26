"""Action samplers.

Action samplers turn pre-distribution parameters (e.g. concatenated mean
and log_std) into a sampled action plus its log-likelihood. They are
ordinary :class:`StatefulModule` s: they live inside the network just
like any other layer, and the user composes them with the other
containers (typically as the last layer of the ``action=`` port of a
:class:`~nnx_ppo.networks.adapter.PPOAdapter`).

Sampler behaviour is driven by ``rollout_extras``:

- ``rollout_extras is None`` (ROLLOUT and INFERENCE): sample fresh. In
  the returned :class:`StatefulModuleOutput.rollout_extras` field, emit
  the sampled raw action so a later LOSS_REPLAY pass can reproduce it.
- ``rollout_extras is not None`` (LOSS_REPLAY): use the stored raw
  action to compute the log-likelihood under the current policy. The
  RNG stream still advances so any downstream stochastic layers stay
  in lockstep with the rollout.

The per-instance ``deterministic`` flag is orthogonal: when set to
``True`` (typically by ``network.eval()``), the sampler returns the
mean instead of sampling. It applies regardless of whether
``rollout_extras`` is provided.

Forward output is a small dict ``{"action", "log_likelihood"}``. The
enclosing :class:`PPOAdapter` lifts each field into the matching dict
on :class:`PPONetworkOutput`. Mean / std live in the sampler's
``metrics`` for logging.
"""

import abc
from typing import Any, Optional

import jax
import jax.numpy as jp
from flax import nnx
from jaxtyping import Array, Float

from nnx_ppo.networks.types import (
    StatefulModule,
    StatefulModuleOutput,
)


class ActionSampler(StatefulModule, abc.ABC):
    deterministic: bool = False

    @abc.abstractmethod
    def __call__(
        self,
        state: tuple[()],
        mean_and_std: Float[Array, "batch mean_std_dim"],
        rollout_extras: Optional[Float[Array, "batch action_dim"]] = None,
    ) -> StatefulModuleOutput:
        """Apply the sampler.

        Args:
            state: Empty tuple (stateless sampler).
            mean_and_std: Concatenated mean and std, shape
                ``[batch, 2 * action_dim]``.
            rollout_extras: ``None`` to sample fresh (ROLLOUT / INFERENCE);
                the stored action to reuse (LOSS_REPLAY).
        """


class NormalTanhSampler(ActionSampler):
    """Normal distribution followed by tanh."""

    def __init__(
        self,
        rng: nnx.Rngs,
        entropy_weight: float,
        min_std: float = 1e-3,
        std_scale: float = 1.0,
    ):
        self.rng = rng
        self.min_std = min_std
        self.std_scale = std_scale
        self.deterministic = False
        self.entropy_weight = entropy_weight

    def __call__(
        self,
        state: tuple[()],
        mean_and_std: Float[Array, "batch mean_std_dim"],
        rollout_extras: Optional[Float[Array, "batch action_dim"]] = None,
    ) -> StatefulModuleOutput:
        mean, std = jp.split(mean_and_std, 2, axis=-1)
        std = (jax.nn.softplus(std) + self.min_std) * self.std_scale

        # Sample even when rollout_extras is supplied, so the RNG stream
        # advances identically across the rollout and the loss replay.
        if self.deterministic:
            sampled_action = mean
        else:
            sampled_action = mean + std * jax.random.normal(self.rng(), mean.shape)

        if rollout_extras is None:
            raw_action = jax.lax.stop_gradient(sampled_action)
        else:
            raw_action = rollout_extras

        action = jp.tanh(raw_action)
        loglikelihood = self._loglikelihood(raw_action, mean, std)
        entropy_cost = -self.entropy_weight * self._entropy(mean, std)

        return StatefulModuleOutput(
            next_state=(),
            output={"action": action, "log_likelihood": loglikelihood},
            regularization_loss=entropy_cost,
            metrics={"mu": mean, "sigma": std},
            rollout_extras=raw_action,
        )

    def initialize_state(self, batch_size: int) -> tuple[()]:
        return ()

    def _loglikelihood(
        self,
        raw_action: Float[Array, "batch action_dim"],
        mean: Float[Array, "batch action_dim"],
        std: Float[Array, "batch action_dim"],
    ) -> Float[Array, "batch"]:
        z = raw_action

        # Normal log-likelihood.
        log_unnormalized = -0.5 * jp.square((z - mean) / std)
        log_normalization = 0.5 * jp.log(2.0 * jp.pi) + jp.log(std)
        log_prob = log_unnormalized - log_normalization

        # Numerically stable log|d/dz tanh(z)| correction (Brax-style).
        log_det_jacobian = 2.0 * (jp.log(2.0) - z - jax.nn.softplus(-2.0 * z))
        log_prob -= log_det_jacobian

        return jp.sum(log_prob, axis=-1)

    def _entropy(
        self,
        mean: Float[Array, "batch action_dim"],
        std: Float[Array, "batch action_dim"],
    ) -> Float[Array, "batch"]:
        normal_entropy = 0.5 + 0.5 * jp.log(2.0 * jp.pi) + jp.log(std)
        z = mean + std * jax.lax.stop_gradient(
            jax.random.normal(self.rng(), mean.shape)
        )
        log_det_jacobian = 2.0 * (jp.log(2.0) - z - jax.nn.softplus(-2.0 * z))
        return jp.sum(normal_entropy + log_det_jacobian, axis=-1)
