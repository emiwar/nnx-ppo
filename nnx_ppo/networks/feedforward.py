from typing import Tuple, List, Callable, Optional

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.sampling_layers import ActionSampler, NormalTanhSampler
from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput
from nnx_ppo.networks.normalizer import Normalizer


class MLP(StatefulModule):
    def __init__(self, sizes: List[int], rngs,
                 transfer_function: Callable = nnx.relu,
                 transfer_function_last_layer: bool=True,
                 params_for_Linear={}):
        din_dout = zip(sizes[:-1], sizes[1:])
        self.layers = nnx.List([nnx.Linear(din, dout, rngs=rngs, **params_for_Linear) for (din, dout) in din_dout])
        self.transfer_function = transfer_function
        self.transfer_function_last_layer = transfer_function_last_layer

    def __call__(self, state: Tuple, x: jax.Array) -> StatefulModuleOutput:
        for layer in self.layers[:-1]:
            x = self.transfer_function(layer(x))
        x = self.layers[-1](x)
        if self.transfer_function_last_layer:
            x = self.transfer_function(x)
        return StatefulModuleOutput(state, x, jp.array(0.0), {})


# Import PPOActorCritic here to avoid circular import (it depends on MLP)
from nnx_ppo.networks.containers import PPOActorCritic


class MLPActorCritic(PPOActorCritic):
    def __init__(self,
                 obs_size: int,
                 action_size: int,
                 actor_hidden_sizes: List[int],
                 critic_hidden_sizes: List[int],
                 rngs: nnx.Rngs,
                 transfer_function: Callable = nnx.relu,
                 action_sampler: Optional[ActionSampler] = None,
                 normalize_obs: bool = False,
                 initalizer_scale = 1.0):
        if action_sampler is None:
          action_sampler = NormalTanhSampler(rngs, entropy_weight=1e-3)
        actor_sizes = [obs_size] + actor_hidden_sizes + [action_size*2]
        self.preprocessor = Normalizer(obs_size) if normalize_obs else None
        self.actor = MLP(actor_sizes, rngs, transfer_function, transfer_function_last_layer=False,
                         params_for_Linear={'kernel_init': nnx.initializers.variance_scaling(initalizer_scale, "fan_in", "uniform")})
        critic_sizes = [obs_size] + critic_hidden_sizes + [1]
        self.critic = MLP(critic_sizes, rngs, transfer_function, transfer_function_last_layer=False,
                          params_for_Linear={'kernel_init': nnx.initializers.variance_scaling(initalizer_scale, "fan_in", "uniform")})
        self.action_sampler = action_sampler
