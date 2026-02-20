from typing import Optional
import abc

import jax
import jax.numpy as jp
from flax import nnx
from jaxtyping import Array, Float

from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput


class ActionSampler(StatefulModule, abc.ABC):

    @abc.abstractmethod
    def __call__(
        self,
        state: tuple[()],
        mean_and_std: Float[Array, "batch mean_std_dim"],
        raw_action: Optional[Float[Array, "batch mean_std_dim//2"]] = None,
    ) -> StatefulModuleOutput:
        """Apply the layer.

        Args:
            state: Empty tuple (stateless sampler).
            mean_and_std: Concatenated mean and std, shape (batch, 2*action_dim).
            raw_action: Optional pre-sampled action, shape (batch, action_dim).
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
        raw_action: Optional[Float[Array, "batch mean_std_dim//2"]] = None,
    ) -> StatefulModuleOutput:
        mean, std = jp.split(mean_and_std, 2, axis=-1)
        # std = self.std_scale*0.5*(jp.tanh(std)+1) + self.min_std
        std = (jax.nn.softplus(std) + self.min_std) * self.std_scale

        # We want to sample an action even if raw_action is specified, so that the
        # state of the RNG is consistent after the call.
        if self.deterministic:
            sampled_action = mean
        else:
            sampled_action = mean + std * jax.random.normal(self.rng(), mean.shape)
        if raw_action is None:
            raw_action = jax.lax.stop_gradient(sampled_action)
        action = jp.tanh(raw_action)
        loglikelihood = self._loglikelihood(raw_action, mean, std)
        entropy_cost = -self.entropy_weight * self._entropy(mean, std)
        return StatefulModuleOutput(
            next_state=(),
            output=(action, raw_action, loglikelihood),
            regularization_loss=entropy_cost,
            metrics={"mu": mean, "sigma": std},
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

        # Log-likelihood for normal:
        log_unnormalized = -0.5 * jp.square((z - mean) / std)
        log_normalization = 0.5 * jp.log(2.0 * jp.pi) + jp.log(std)
        log_prob = log_unnormalized - log_normalization

        # Modify the log-likelihood due to tanh transformation. Should be log|d/dz tanh(z)|.
        # The expression below is a numerically stable version of log|d/dz tanh(z)| borrowed from Brax
        log_det_jacobian = 2.0 * (jp.log(2.0) - z - jax.nn.softplus(-2.0 * z))
        log_prob -= log_det_jacobian

        # Sum over last dimension if needed
        log_prob = jp.sum(log_prob, axis=-1)

        return log_prob

    def _entropy(
        self,
        mean: Float[Array, "batch action_dim"],
        std: Float[Array, "batch action_dim"],
    ) -> Float[Array, "batch"]:
        # Entropy per dimension, sum over action dimensions
        normal_entropy = 0.5 + 0.5 * jp.log(2.0 * jp.pi) + jp.log(std)
        # No analytical formula for entropy, use a single Monte Carlo sample
        z = mean + std * jax.lax.stop_gradient(
            jax.random.normal(self.rng(), mean.shape)
        )
        log_det_jacobian = 2.0 * (jp.log(2.0) - z - jax.nn.softplus(-2.0 * z))
        return jp.sum(normal_entropy + log_det_jacobian, axis=-1)
