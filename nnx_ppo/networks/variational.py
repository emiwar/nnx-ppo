from typing import Any

import jax
import jax.numpy as jp
from flax import nnx
from jaxtyping import Array, Float, Key, PRNGKeyArray

from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput


class VariationalBottleneck(StatefulModule):
    """Variational bottleneck with KL regularization.

    Takes input of size 2*latent_size (mean and log_std concatenated),
    samples from the latent distribution using the reparameterization trick,
    and adds KL divergence against a standard normal prior to the regularization loss.
    """

    def __init__(
        self, latent_size: int, rng, kl_weight: float = 1.0, min_std: float = 1e-6
    ):
        """Initialize the variational bottleneck.

        Args:
            latent_size: Size of the latent space. Input is expected to be 2*latent_size.
            rngs: NNX random number generators.
            kl_weight: Weight for the KL divergence regularization term.
            min_std: Minimum standard deviation to prevent numerical instability.
        """
        self.latent_size = latent_size
        self.rng = rng
        self.kl_weight = kl_weight
        self.min_std = min_std

    def __call__(
        self, key: Key[Array, "batch"], x: Float[Array, "batch {2*self.latent_size}"]
    ) -> StatefulModuleOutput:
        """Sample from the variational distribution.

        Args:
            key: rng key.
            x: Input array of shape (..., 2*latent_size) containing concatenated
               mean and log_std.

        Returns:
            StatefulModuleOutput with:
                - next_state: RNG keys
                - output: Sampled latent vector of shape (..., latent_size)
                - regularization_loss: KL divergence weighted by kl_weight
                - metrics: Dictionary with mu, sigma, and kl_divergence
        """
        # Split input into mean and log_std
        mean, log_std = jp.split(x, 2, axis=-1)
        std = jax.nn.softplus(log_std) + self.min_std

        # Sample using reparameterization trick
        eps = jax.vmap(lambda k: jax.random.normal(k, (self.latent_size,)))(key)
        z = mean + std * eps

        # KL divergence against standard normal: KL(N(mu, sigma) || N(0, 1))
        # = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
        kl_per_dim = 0.5 * (jp.square(mean) + jp.square(std) - 2 * jp.log(std) - 1)
        kl_divergence = jp.sum(kl_per_dim, axis=-1)  # Sum over latent dimensions
        kl_loss = self.kl_weight * kl_divergence

        next_key, _ = jax.vmap(jax.random.split, out_axes=1)(key)

        return StatefulModuleOutput(
            next_state=next_key,
            output=z,
            regularization_loss=kl_loss,
            metrics={
                "mu": mean,
                "sigma": std,
                "kl_divergence": kl_divergence,
            },
        )

    def initialize_state(self, batch_size: int) -> Key[Array, "batch"]:
        return jax.random.split(self.rng(), batch_size)

    def reset_state(self, prev_state: Key[Array, "batch"]) -> Key[Array, "batch"]:
        # It's fine to keep the chain of rng keys across env resets
        return prev_state


# AR1VariationalBottleneck state: Dict with 'keys' (PRNGKeyArray) and 'last_z'
# (Float[batch, latent])
AR1State = dict[str, Any]


class AR1VariationalBottleneck(StatefulModule):
    """Variational bottleneck with KL and auto-regressive regularization.

    Takes input of size 2*latent_size (mean and log_std concatenated),
    samples from the latent distribution using the reparameterization trick,
    adds KL divergence against a standard normal prior to the regularization loss,
    and adds a first-order autoregressive loss. The autoregressive loss encourages
    smoother trajectories in the latent space.
    """

    def __init__(
        self,
        latent_size: int,
        rng,
        kl_weight: float = 1.0,
        min_std: float = 1e-6,
        ar1_weight: float = 1.0,
        backprop_through_time: bool = True,
    ):
        """Initialize the variational bottleneck.

        Args:
            latent_size: Size of the latent space. Input is expected to be 2*latent_size.
            rngs: NNX random number generators.
            kl_weight: Weight for the KL divergence regularization term.
            min_std: Minimum standard deviation to prevent numerical instability.
            ar1_weight: Weight for the autoregressive loss
            backprop_through_time: Conceptually, the AR1 loss can be minimized in two
                ways: either by making the current latent vector (z) closer to the previous
                latent vector (prev_z), or by making prev_z closer to z. The latter
                requires the gradient to flow back in time, which is perfectly fine
                in most instances. However, this param can be set to `False` to turn
                that off.
        """
        self.latent_size = latent_size
        self.rng = rng
        self.kl_weight = kl_weight
        self.min_std = min_std
        self.ar1_weight = ar1_weight
        self.backprop_through_time = backprop_through_time

    def __call__(
        self,
        state: AR1State,
        x: Float[Array, "batch {2*self.latent_size}"],
    ) -> StatefulModuleOutput:
        """Sample from the variational distribution.

        Args:
            state: Dict with 'keys' (PRNGKeyArray) and 'last_z' (Float[batch, latent]).
            x: Input array of shape (batch, 2*latent) containing concatenated
               mean and log_std.

        Returns:
            StatefulModuleOutput with:
                - next_state: RNG keys and prev_z
                - output: Sampled latent vector of shape (batch, latent)
                - regularization_loss: sum of KL loss and AR1 loss
                - metrics: Dictionary with mu, sigma, kl_divergence, and squared diff
        """
        keys = state["keys"]
        prev_z: Float[Array, "batch {self.latent_size}"] = state["last_z"]
        if not self.backprop_through_time:
            prev_z = jax.lax.stop_gradient(prev_z)

        # Split input into mean and log_std
        mean, log_std = jp.split(x, 2, axis=-1)
        std = jax.nn.softplus(log_std) + self.min_std

        # Sample using reparameterization trick
        eps = jax.vmap(lambda k: jax.random.normal(k, (self.latent_size,)))(keys)
        z = mean + std * eps

        # KL divergence against standard normal: KL(N(mu, sigma) || N(0, 1))
        # = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
        kl_per_dim = 0.5 * (jp.square(mean) + jp.square(std) - 2 * jp.log(std) - 1)
        kl_divergence = jp.sum(kl_per_dim, axis=-1)  # Sum over latent dimensions
        kl_loss = self.kl_weight * kl_divergence

        # AR1 loss
        # Replace NaN in prev_z with z so that (z - safe_prev_z) = 0 when prev_z is NaN.
        # This avoids NaN gradients during backprop while also making l2_diff = 0.
        safe_prev_z = jp.where(jp.isnan(prev_z), z, prev_z)
        l2_diff = jp.mean(jp.square(z - safe_prev_z), axis=-1)
        ar1_loss = self.ar1_weight * l2_diff

        total_regularization_loss = kl_loss + ar1_loss

        next_keys, _ = jax.vmap(jax.random.split, out_axes=1)(keys)
        next_state = {
            "keys": next_keys,
            "last_z": z,
        }

        return StatefulModuleOutput(
            next_state=next_state,
            output=z,
            regularization_loss=total_regularization_loss,
            metrics={
                "mu": mean,
                "sigma": std,
                "kl_divergence": kl_divergence,
                "l2_diff": l2_diff,
            },
        )

    def initialize_state(self, batch_size: int) -> AR1State:
        return {
            "keys": jax.random.split(self.rng(), batch_size),
            "last_z": jp.full((batch_size, self.latent_size), jp.nan),
        }

    def reset_state(self, prev_state: AR1State) -> AR1State:
        # It's fine to keep the chain of rng keys across env resets but last_z
        # should be set to NaN
        return {
            "keys": prev_state["keys"],
            "last_z": jp.full_like(prev_state["last_z"], jp.nan),
        }
