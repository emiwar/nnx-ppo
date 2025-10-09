from typing import Tuple

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import StatefulModule

class ActionSampler(StatefulModule):
  pass

class NormalTanhSampler(ActionSampler):
  """Normal distribution followed by tanh."""

  def __init__(self, rng: nnx.Rngs, entropy_weight: float, min_std: float=1e-3, std_scale:float=1.0, preclamp: bool=True):
    self.rng = rng
    self.min_std = min_std
    self.std_scale = std_scale
    self.preclamp = preclamp
    self.deterministic = False
    self.entropy_weight = entropy_weight

  def __call__(self, state, mean_and_std: jax.Array) -> Tuple[Tuple[()], Tuple[jax.Array, jax.Array], jax.Array]:
    if self.preclamp:
      # The sampling will clamp the actions to be in (-1, 1), but we might also want
      # clamp the mean and std themselves.
      mean_and_std = jp.tanh(mean_and_std)
    mean, std = jp.split(mean_and_std, 2, axis=-1)
    std = (jax.nn.softplus(std) + self.min_std) * self.std_scale
    if self.deterministic:
      raw_action = mean
    else:
      raw_action = mean + std * jax.random.normal(self.rng(), mean.shape)
    raw_action = jax.lax.stop_gradient(raw_action)
    action = jp.tanh(raw_action)
    loglikelihood = self._loglikelihood(raw_action, mean, std)
    entropy_cost = -self.entropy_weight * self._entropy(std)
    return (), (action, loglikelihood), entropy_cost
  
  def initialize_state(self, batch_size: int) -> Tuple[()]:
    return ()

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
  
  def _entropy(self, std):
    return 0.5 + 0.5 * jp.log(2.0 * jp.pi) + jp.log(std)
