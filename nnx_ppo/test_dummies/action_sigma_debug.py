import jax
import jax.numpy as jp
from flax import nnx

def extract_sigma(networks, obs):
    actor_state, actor_output, actor_reg = networks.actor((), obs)
    raw_mean, std = jp.split(actor_output, 2, axis=-1)
    action_sigma = (jax.nn.softplus(std) + networks.action_sampler.min_std) * networks.action_sampler.std_scale
    return action_sigma, raw_mean

def extra_metrics(networks, rollout):
    action_sigmas, raw_mean = nnx.vmap(nnx.vmap(extract_sigma, in_axes=(None, 0)), in_axes=(None, 0))(networks, rollout.obs)
    return {"action_sigmas/mean": action_sigmas.mean(),
            "action_sigmas/std":  action_sigmas.std(),
            "action_sigmas/min":  action_sigmas.min(),
            "action_sigmas/max":  action_sigmas.max(),
            "action_mus/mean": raw_mean.mean(),
            "action_mus/std":  raw_mean.std(),
            "action_mus/min":  raw_mean.min(),
            "action_mus/max":  raw_mean.max()}
    