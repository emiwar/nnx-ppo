from typing import Tuple

import jax
import jax.numpy as jp

from nnx_ppo.networks.types import StatefulModule

class ActionSampler(StatefulModule):
  pass

class NormalTanhSampler(ActionSampler):
  """Normal distribution followed by tanh."""

  def __init__(self, entropy_weight: float, min_std: float=1e-3, std_scale:float=1.0):
    #elf.rng = rngs.action_sampling
    self.min_std = min_std
    self.std_scale = std_scale
    self.deterministic = False
    self.entropy_weight = entropy_weight


  def __call__(self, rng_key, mean_and_std: jax.Array) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array], jax.Array]:
    action_rng_key, new_rng_key = jax.random.split(rng_key)
    mean, std = jp.split(mean_and_std, 2, axis=-1)
    mean = 0.005 + jp.zeros_like(mean) #0.0 #0.01
    std = 0.25#(jax.nn.softplus(std) + self.min_std) * self.std_scale
    #std = (jp.square(std) + self.min_std) * self.std_scale
    if self.deterministic:
      raw_action = mean
    else:
      raw_action = mean + std * jax.random.normal(action_rng_key, mean.shape) 
    action = jp.tanh(raw_action)
    loglikelihood = self._loglikelihood(raw_action, mean, std)
    # Q: Should entropy cost be negative?
    entropy_cost = self.entropy_weight * self._entropy(std)
    return new_rng_key, (action, loglikelihood), entropy_cost
  
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
  
  def _entropy(self, std):
    return 0.5 + 0.5 * jp.log(2.0 * jp.pi) + jp.log(std)
  
class NormalSampler(ActionSampler):
  """Normal distribution."""

  def __init__(self, entropy_weight: float, min_std: float=1e-3, std_scale:float=1.0):
    #elf.rng = rngs.action_sampling
    self.min_std = min_std
    self.std_scale = std_scale
    self.deterministic = False
    self.entropy_weight = entropy_weight


  def __call__(self, rng_key, mean_and_std: jax.Array) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array], jax.Array]:
    action_rng_key, new_rng_key = jax.random.split(rng_key)
    mean, std = jp.split(mean_and_std, 2, axis=-1)
    std = (jax.nn.softplus(std) + self.min_std) * self.std_scale
    #std = (jp.square(std) + self.min_std) * self.std_scale
    if self.deterministic:
      action = mean
    else:
      action = mean + std * jax.random.normal(action_rng_key, mean.shape)
    loglikelihood = self._loglikelihood(action, mean, std)
    # Q: Should entropy cost be negative?
    entropy_cost = self.entropy_weight * self._entropy(std)
    return new_rng_key, (action, loglikelihood), entropy_cost
  
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

    # Sum over last dimension if needed
    log_prob = jp.sum(log_prob, axis=-1)
    
    return log_prob
  
  def _entropy(self, std):
    return 0.5 + 0.5 * jp.log(2.0 * jp.pi) + jp.log(std)