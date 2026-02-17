from typing import Tuple, Optional

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput


class VariationalBottleneck(StatefulModule):
    """Variational bottleneck with KL regularization.

    Takes input of size 2*latent_size (mean and log_std concatenated),
    samples from the latent distribution using the reparameterization trick,
    and adds KL divergence against a standard normal prior to the regularization loss.
    """

    def __init__(self, latent_size: int, seed: int, kl_weight: float = 1.0, min_std: float = 1e-6):
        """Initialize the variational bottleneck.

        Args:
            latent_size: Size of the latent space. Input is expected to be 2*latent_size.
            rngs: NNX random number generators.
            kl_weight: Weight for the KL divergence regularization term.
            min_std: Minimum standard deviation to prevent numerical instability.
        """
        self.latent_size = latent_size
        self.seed = seed
        self.kl_weight = kl_weight
        self.min_std = min_std

    def __call__(self, key: jax.Array, x: jax.Array) -> StatefulModuleOutput:
        """Sample from the variational distribution.

        Args:
            key: rng key.
            x: Input array of shape (..., 2*latent_size) containing concatenated
               mean and log_std.

        Returns:
            StatefulModuleOutput with:
                - next_state: Empty tuple (stateless module)
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
        kl_loss = self.kl_weight * kl_divergence  # Mean over batch

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

    def initialize_state(self, batch_size: int) -> jax.Array:
        return jax.random.split(jax.random.key(self.seed), batch_size)
    
    def reset_state(self, prev_state: jax.Array) -> jax.Array:
        # It's fine to keep the chain of rng keys across env resets
        return prev_state
