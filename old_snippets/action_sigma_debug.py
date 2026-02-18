import jax
import jax.numpy as jp
from flax import nnx
from nnx_ppo.algorithms import ppo

def extract_sigma(networks, obs):
    actor_output = networks.actor((), obs).output
    raw_mean, std = jp.split(actor_output, 2, axis=-1)
    action_sigma = (jax.nn.softplus(std) + networks.action_sampler.min_std) * networks.action_sampler.std_scale
    return action_sigma, raw_mean

def extra_metrics(networks, rollout, percentile_levels=None):
    action_sigmas, raw_mean = nnx.vmap(nnx.vmap(extract_sigma, in_axes=(None, 0)), in_axes=(None, 0))(networks, rollout.obs)
    m = dict()
    ppo._log_metric(m, "actor/mu", raw_mean, percentile_levels)
    ppo._log_metric(m, "actor/sigma", action_sigmas, percentile_levels)
    return m