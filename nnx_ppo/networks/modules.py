from typing import Tuple, Dict, List, Callable, Optional

import jax
import jax.flatten_util
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.sampling_layers import ActionSampler, NormalTanhSampler
from nnx_ppo.networks.types import PPONetwork, PPONetworkOutput, StatefulModule, ModuleState

class PPOActorCritic(PPONetwork, nnx.Module):
    '''A general PPO actor-critic network consisting of separate actor and critic
       networks.'''

    def __init__(self, actor: StatefulModule, critic: StatefulModule, action_sampler: ActionSampler, flatten_obs: bool = False):
         self.actor = actor
         self.critic = critic
         self.action_sampler = action_sampler
         self.flatten_obs = flatten_obs

    def __call__(self, network_state, obs) -> Tuple[Dict, PPONetworkOutput]:
        if self.flatten_obs:
            obs = jax.vmap(lambda obs: jax.flatten_util.ravel_pytree(obs)[0], (0,))(obs)
        actor_output = self.actor(network_state["actor"], obs)
        network_state["actor"], sampler_input, actor_reg = actor_output
        sampler_output = self.action_sampler(network_state["action_sampler"], sampler_input)
        network_state["action_sampler"], (action, loglikelihood), sampler_reg = sampler_output
        critic_output = self.critic(network_state["critic"], obs)
        network_state["critic"], critic_output, critic_reg = critic_output
        total_reg_loss = actor_reg + sampler_reg + critic_reg
        return network_state, PPONetworkOutput(
            actions = action,
            loglikelihoods = loglikelihood,
            regularization_loss = total_reg_loss,
            value_estimates = critic_output,
            metrics = dict()
        )
    
    def initialize_state(self, batch_size: int) -> ModuleState:
        return {
             "actor": self.actor.initialize_state(batch_size),
             "critic": self.critic.initialize_state(batch_size),
             "action_sampler": self.action_sampler.initialize_state(batch_size),
        }

class MLP(StatefulModule):
    def __init__(self, sizes, rngs,
                 transfer_function=nnx.relu,
                 transfer_function_last_layer: bool=True):
        din_dout = zip(sizes[:-1], sizes[1:])
        self.layers = [nnx.Linear(din, dout, rngs=rngs) for (din, dout) in din_dout]
        self.transfer_function = transfer_function
        self.transfer_function_last_layer = transfer_function_last_layer

    def __call__(self, state: Tuple, x: jax.Array):
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
                 transfer_function: Callable = nnx.relu,
                 action_sampler: Optional[ActionSampler] = None):
        if action_sampler is None:
          action_sampler = NormalTanhSampler(rngs, entropy_weight=1e-3)
        obs_size = int(jp.sum(jax.flatten_util.ravel_pytree(obs_size)[0]))
        action_size = int(jp.sum(jax.flatten_util.ravel_pytree(action_size)[0]))
        actor_sizes = [obs_size] + actor_hidden_sizes + [action_size*2]
        self.actor = MLP(actor_sizes, rngs, transfer_function, transfer_function_last_layer=False)
        critic_sizes = [obs_size] + critic_hidden_sizes + [1]
        self.critic = MLP(critic_sizes, rngs, transfer_function, transfer_function_last_layer=False)
        self.action_sampler = action_sampler
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
    
    def initialize_state(self, batch_size) -> List:
        state = []
        for layer in self.layers:
            state.append(layer.initialize_state(batch_size))
        return state
    
    def __getitem__(self, ind) -> StatefulModule:
        return self.layers[ind]