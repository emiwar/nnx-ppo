from typing import Tuple

import jax
import jax.numpy as jp

from nnx_ppo.networks.types import StatefulModule

class ActionSampler(StatefulModule):
  pass

class NormalTanhSampler(ActionSampler):
  """Normal distribution followed by tanh."""

  def __init__(self, rngs, min_std=1e-3, std_scale=1.0):
    self.rng = rngs.action_sampling
    self.min_std = min_std
    self.std_scale = std_scale
    self.deterministic = False

  def __call__(self, rng_key, mean_and_std: jax.Array) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array], float]:
    action_rng_key, new_rng_key = jax.random.split(rng_key)
    mean, std = jp.split(mean_and_std, 2, axis=-1)
    std = (jax.nn.softplus(std) + self.min_std) * self.std_scale
    raw_action = mean + std * jax.random.normal(action_rng_key, mean.shape) 
    action = jp.tanh(raw_action)
    loglikelihood = self._loglikelihood(raw_action, mean, std)
    return new_rng_key, (action, loglikelihood), 0.0
  
  def initialize_state(self, rng: jax.Array) -> jax.Array:
    return rng

  def reset_state(self, network_state: jax.Array) -> jax.Array:
    # State is just a key; it doesn't have to be reset
    return network_state

  def _loglikelihood(self, raw_action, mean, std) -> jax.Array:
    z = raw_action

    # Log-likelihood for normal:
    log_unnormalized = -0.5 * jp.square((z - mean) / std)
    log_normalization = 0.5 * jp.log(2.0 * jp.pi) + jp.log(std)
    log_prob = log_unnormalized - log_normalization

    # Modify the log-likelihood due to tanh transformation. Should be log|d/dx tanh(z)|.
    # The expression below is a numerically stable version of log|d/dx tanh(z)| borrowed from Brax
    log_det_jacobian = 2.0 * (jp.log(2.0) - z - jax.nn.softplus(-2.0 * z))
    log_prob -= log_det_jacobian

    # Sum over last dimension if needed
    log_prob = jp.sum(log_prob, axis=-1)
    
    return log_prob