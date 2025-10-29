from typing import Tuple

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput

class ActionSampler(StatefulModule):
  pass

class NormalTanhSampler(ActionSampler):
  """Normal distribution followed by tanh."""

  def __init__(self, rng: nnx.Rngs, entropy_weight: float, min_std: float=1e-3, std_scale:float=1.0, preclamp: bool=False):
    self.rng = rng
    self.min_std = min_std
    self.std_scale = std_scale
    self.preclamp = preclamp
    self.deterministic = False
    self.entropy_weight = entropy_weight

  def __call__(self, state, mean_and_std: jax.Array) -> StatefulModuleOutput:
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

    # Here is an unusual trick. Rather than passing the aciton as a parameter
    # to this layer, we compute it again. This is possible because the RNGStream
    # `self.rng` is reset between rollout and the loss computation, so the same
    # action will be sampled in both passes. Having access to the raw action is
    # numerically more stable than having only the action after the tanh, which
    # could saturate and prevent us from applying a numerically stable inverse. But
    # it's important that the gradient doesn't flow back through the sampling -- it
    # should only flow through the log likelihood. Therefore, we need a stop_gradient.
    raw_action = jax.lax.stop_gradient(raw_action)
    action = jp.tanh(raw_action)
    loglikelihood = self._loglikelihood(raw_action, mean, std)
    entropy_cost = -self.entropy_weight * self._entropy(std)
    return StatefulModuleOutput(
      next_state=(),
      output=(action, loglikelihood),
      regularization_loss=entropy_cost,
      metrics=dict(),
    )
  
  def initialize_state(self, batch_size: int) -> Tuple[()]:
    return ()

  def _loglikelihood(self, raw_action, mean, std) -> jax.Array:
    z = raw_action

    # Log-likelihood for normal:
    log_unnormalized = -0.5 * jp.square((z - mean) / std)
    log_normalization = 0.5 * jp.log(2.0 * jp.pi) + jp.log(std)
    log_prob = log_unnormalized - log_normalization

    # Modify the log-likelihood due to tanh transformation. Should be log|d/dz tanh(z)|.
    # The expression below is a numerically stable version of log|d/dz tanh(z)| borrowed from Brax
    log_det_jacobian = 2.0 * (jp.log(2.0) - z - jax.nn.softplus(-2.0 * z))
    log_prob += log_det_jacobian

    # Sum over last dimension if needed
    log_prob = jp.sum(log_prob, axis=-1)
    
    return log_prob
  
  def _entropy(self, std):
    return 0.5 + 0.5 * jp.log(2.0 * jp.pi) + jp.log(std)
