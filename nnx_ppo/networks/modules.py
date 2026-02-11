from typing import Any, Tuple, Dict, List, Callable, Optional, Union

import jax
import jax.flatten_util
import jax.numpy as jp
from flax import nnx

from nnx_ppo.algorithms.types import Transition
from nnx_ppo.networks.sampling_layers import ActionSampler, NormalTanhSampler
from nnx_ppo.networks.types import PPONetwork, PPONetworkOutput, StatefulModule, ModuleState, StatefulModuleOutput

class PPOActorCritic(PPONetwork, nnx.Module):
    '''A general PPO actor-critic network consisting of separate actor and critic
       networks.'''

    def __init__(self, actor: StatefulModule, critic: StatefulModule, action_sampler: ActionSampler, preprocessor: Optional[StatefulModule] = None):
         self.actor = actor
         self.critic = critic
         self.action_sampler = action_sampler
         self.preprocessor = preprocessor

    def __call__(self, network_state, obs, raw_action: Optional[jax.Array] = None) -> Tuple[Dict, PPONetworkOutput]:
        #if self.flatten_obs:
        #    obs = jax.vmap(lambda obs: jax.flatten_util.ravel_pytree(obs)[0], (0,))(obs)
        regularization_loss = jp.array(0.0)
        if self.preprocessor is not None:
            preprocessor_output = self.preprocessor(network_state["preprocessor"], obs)
            obs = preprocessor_output.output
            network_state["preprocessor"] = preprocessor_output.next_state
            regularization_loss += preprocessor_output.regularization_loss
        actor_output = self.actor(network_state["actor"], obs)
        sampler_output = self.action_sampler(network_state["action_sampler"], actor_output.output, raw_action)
        action, raw_action, loglikelihood = sampler_output.output
        critic_output = self.critic(network_state["critic"], obs)

        network_state["actor"] = actor_output.next_state
        network_state["action_sampler"] = sampler_output.next_state
        network_state["critic"] = critic_output.next_state
        regularization_loss += actor_output.regularization_loss
        regularization_loss += sampler_output.regularization_loss
        regularization_loss += critic_output.regularization_loss
        return network_state, PPONetworkOutput(
            actions = action,
            raw_actions = raw_action,
            loglikelihoods = loglikelihood,
            regularization_loss = regularization_loss,
            value_estimates = critic_output.output,
            metrics = {
                "preprocessor": preprocessor_output.metrics if self.preprocessor is not None else {},
                "actor": actor_output.metrics,
                "critic": critic_output.metrics,
                "action_sampler": sampler_output.metrics,
            }
        )
    
    def initialize_state(self, batch_size: int) -> ModuleState:
        return {
             "preprocessor": self.preprocessor.initialize_state(batch_size) if self.preprocessor is not None else (),
             "actor": self.actor.initialize_state(batch_size),
             "critic": self.critic.initialize_state(batch_size),
             "action_sampler": self.action_sampler.initialize_state(batch_size),
        }
    
    def update_statistics(self, last_rollout: Transition, total_steps: jax.Array) -> None:
        if self.preprocessor is not None:
            self.preprocessor.update_statistics(last_rollout, total_steps)
        self.actor.update_statistics(last_rollout, total_steps)
        self.critic.update_statistics(last_rollout, total_steps)
        self.action_sampler.update_statistics(last_rollout, total_steps)


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

class MLPActorCritic(PPOActorCritic):
    def __init__(self,
                 obs_size: int,
                 action_size: int,
                 actor_hidden_sizes: List[int],
                 critic_hidden_sizes: List[int],
                 rngs: nnx.Rngs,
                 transfer_function: Callable = nnx.relu,
                 action_sampler: Optional[ActionSampler] = None,
                 normalize_obs: bool = False):
        if action_sampler is None:
          action_sampler = NormalTanhSampler(rngs, entropy_weight=1e-3)
        actor_sizes = [obs_size] + actor_hidden_sizes + [action_size*2]
        self.preprocessor = Normalizer(obs_size) if normalize_obs else None
        self.actor = MLP(actor_sizes, rngs, transfer_function, transfer_function_last_layer=False,
                         params_for_Linear={'kernel_init': nnx.initializers.lecun_normal()})
        critic_sizes = [obs_size] + critic_hidden_sizes + [1]
        self.critic = MLP(critic_sizes, rngs, transfer_function, transfer_function_last_layer=False,
                          params_for_Linear={'kernel_init': nnx.initializers.lecun_normal()})
        #'kernel_init': nnx.initializers.lecun_uniform()})
        self.action_sampler = action_sampler
        #self.flatten_obs = True

class Sequential(StatefulModule):
    def __init__(self, layers: List[StatefulModule]):
        self.layers = nnx.List(layers)

    def __call__(self, network_state: List, obs) -> StatefulModuleOutput:
        new_network_state = []
        x = obs
        regularization_loss = jp.array(0.0)
        for layer, layer_state in zip(self.layers, network_state):
            layer_output = layer(layer_state, x)
            new_state = layer_output.next_state
            x = layer_output.output
            new_network_state.append(new_state)
            regularization_loss += layer_output.regularization_loss
        return StatefulModuleOutput(new_network_state, x, regularization_loss, {})
    
    def initialize_state(self, batch_size) -> List:
        state = []
        for layer in self.layers:
            state.append(layer.initialize_state(batch_size))
        return state
    
    def __getitem__(self, ind) -> StatefulModule:
        return self.layers[ind]
    
    def update_statistics(self, last_rollout: Transition, total_steps: jax.Array) -> None:
        for layer in self.layers:
            layer.update_statistics(last_rollout, total_steps)
    
class NormalizerStatistics(nnx.Variable): pass

class Normalizer(StatefulModule):

    def __init__(self, shape):
        if isinstance(shape, (tuple, list, int)):
            self.mean = NormalizerStatistics(jp.zeros(shape))
            self.M2 = NormalizerStatistics(jp.zeros(shape))
        else:
            self.mean = NormalizerStatistics(jax.tree.map(jp.zeros_like, shape))
            self.M2 = NormalizerStatistics(jax.tree.map(jp.zeros_like, shape))
        self.counter = NormalizerStatistics(jp.array(0.0))
        self.epsilon = 1e-6

    def __call__(self, state, x):
        # Compute variance from M2
        std = jax.lax.cond(self.counter.value>0,
                           self.M2_to_std,
                           lambda M2: jax.tree.map(lambda x: jp.full(x.shape, 10.0), M2),
                           self.M2.value)
        output = jax.tree.map(lambda x, m, s: (x-m)/s, x, self.mean.value, std)
        return StatefulModuleOutput(
            next_state = (),
            output = output,
            regularization_loss = jp.array(0.0),
            metrics={}
        )

    def M2_to_std(self, M2):
        return jax.tree.map(lambda x: jp.sqrt(jp.maximum(x/self.counter, self.epsilon)), M2)
    
    def update_statistics(self, last_rollout: Transition, total_steps: jax.Array) -> None:
        obs = last_rollout.obs
        batch_count = last_rollout.done.size
        new_count = self.counter.value + batch_count
        frac = batch_count / new_count

        # Welford's algorithm for batched updates
        batch_mean = jax.tree.map(lambda x: jp.mean(x, axis=(0, 1)), obs)
        delta_old = jax.tree.map(lambda batch_mean, old: batch_mean-old, batch_mean, self.mean.value)
        self.mean.value = jax.tree.map(lambda old, delta: old + delta * frac, self.mean.value, delta_old)
        delta_new = jax.tree.map(lambda batch_mean, old: batch_mean-old, batch_mean, self.mean.value)

        batch_var = jax.tree.map(lambda x: jp.var(x, axis=(0, 1)), obs)
        self.M2.value = jax.tree.map(lambda old, batch_var, delta_old, delta_new: old + batch_count*batch_var + batch_count*delta_old*delta_new, 
                                     self.M2.value, batch_var, delta_old, delta_new)

        # Update counter
        self.counter.value += batch_count
