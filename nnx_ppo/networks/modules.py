from typing import Tuple, Dict, List, Callable

import jax
import jax.flatten_util
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.sampling_layers import ActionSampler, NormalTanhSampler
from nnx_ppo.networks.types import AbstractPPOActorCritic, PPONetworkOutput, StatefulModule

class PPOActorCritic(AbstractPPOActorCritic, nnx.Module):

    def __init__(self, actor: StatefulModule, critic: StatefulModule, action_sampler: ActionSampler, flatten_obs: bool = False):
         self.actor = actor
         self.critic = critic
         self.action_sampler = action_sampler
         self.flatten_obs = flatten_obs

    def __call__(self, network_state, obs) -> Tuple[Dict, PPONetworkOutput]:
        if self.flatten_obs:
            obs, _ = jax.flatten_util.ravel_pytree(obs)
        network_state["actor"], actor_output, actor_reg = self.actor(network_state["actor"], obs)
        network_state["action_sampler"], (action, loglikelihood), sampler_reg = self.action_sampler(network_state["action_sampler"], actor_output)
        network_state["critic"], critic_output, critic_reg = self.critic(network_state["critic"], obs)
        total_reg_loss = actor_reg + sampler_reg + critic_reg
        return network_state, PPONetworkOutput(
            actions = action,
            loglikelihoods = loglikelihood,
            regularization_loss = total_reg_loss,
            value_estimates = critic_output,
            metrics = dict()
        )
    
    def initialize_state(self, rng: jax.Array):
        actor_rng, critic_rng, sampler_rng = jax.random.split(rng, 3)
        return {
             "actor": self.actor.initialize_state(actor_rng),
             "critic": self.critic.initialize_state(critic_rng),
             "action_sampler": self.action_sampler.initialize_state(sampler_rng),
        }
    
    def reset_state(self, network_state):
        return {
             "actor": self.actor.reset_state(network_state["actor"]),
             "critic": self.critic.reset_state(network_state["critic"]),
             "action_sampler": self.action_sampler.reset_state(network_state["action_sampler"]),
        }

class MLP(StatefulModule):
    def __init__(self, sizes, rngs,
                 transfer_function=nnx.softplus,
                 transfer_function_last_layer: bool=False):
        din_dout = zip(sizes[:-1], sizes[1:])
        self.layers = [nnx.Linear(din, dout, rngs=rngs) for (din, dout) in din_dout]
        self.transfer_function = transfer_function
        self.transfer_function_last_layer = transfer_function_last_layer

    def __call__(self, state, x):
        for layer in self.layers[:-1]:
            x = self.transfer_function(layer(x))
        x = self.layers[-1](x)
        if self.transfer_function_last_layer:
            x = self.transfer_function(x)
        regularization_loss = 0.0
        return state, x, regularization_loss

class MLPActorCritic(PPOActorCritic):
    def __init__(self,
                 obs_size,
                 action_size,
                 actor_hidden_sizes: List[int],
                 critic_hidden_sizes: List[int],
                 rngs: nnx.Rngs,
                 transfer_function: Callable = nnx.softplus,
                 entropy_weight: float =1e-4):
        obs_size = int(jp.sum(jax.flatten_util.ravel_pytree(obs_size)[0]))
        action_size = int(jp.sum(jax.flatten_util.ravel_pytree(action_size)[0]))
        actor_sizes = [obs_size] + actor_hidden_sizes + [action_size*2]
        self.actor = MLP(actor_sizes, rngs, transfer_function)
        critic_sizes = [obs_size] + critic_hidden_sizes + [1]
        self.critic = MLP(critic_sizes, rngs, transfer_function, transfer_function_last_layer=False)
        self.action_sampler = NormalTanhSampler(entropy_weight=entropy_weight)
        self.flatten_obs = True

class Sequential(StatefulModule):
    def __init__(self, layers: List[StatefulModule]):
        self.layers = layers

    def __call__(self, network_state: List, obs):
        new_network_state = []
        x = obs
        regularization_loss = 0.0
        for layer, layer_state in zip(self.layers, network_state):
            new_state, x, layer_reg_loss = layer(layer_state, x)
            new_network_state.append(new_state)
            regularization_loss += layer_reg_loss
        return new_network_state, x, regularization_loss
    
    def initialize_state(self, rng: jax.Array) -> List:
        state = []
        rngs = jax.random.split(rng, len(self.layers))
        for layer, layer_rng in zip(self.layers, rngs):
            state.append(layer.initialize_state(layer_rng))
        return state

    def reset_state(self, network_state: List):
        new_state = []
        for layer, old_state in zip(self.layers, network_state):
            new_state.append(layer.reset_state(old_state))
        return new_state
    
    def __getitem__(self, ind) -> StatefulModule:
        return self.layers[ind]