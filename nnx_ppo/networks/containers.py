from typing import Any, Tuple, Dict, List, Optional

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.algorithms.types import Transition
from nnx_ppo.networks.sampling_layers import ActionSampler
from nnx_ppo.networks.types import (
    PPONetwork,
    PPONetworkOutput,
    StatefulModule,
    ModuleState,
    StatefulModuleOutput,
)


class PPOActorCritic(PPONetwork, nnx.Module):
    """A general PPO actor-critic network consisting of separate actor and critic
    networks."""

    def __init__(
        self,
        actor: StatefulModule,
        critic: StatefulModule,
        action_sampler: ActionSampler,
        preprocessor: Optional[StatefulModule] = None,
    ):
        self.actor = actor
        self.critic = critic
        self.action_sampler = action_sampler
        self.preprocessor = preprocessor

    def __call__(
        self, network_state, obs, raw_action: Optional[jax.Array] = None
    ) -> Tuple[Dict, PPONetworkOutput]:
        regularization_loss = jp.array(0.0)
        if self.preprocessor is not None:
            preprocessor_output = self.preprocessor(network_state["preprocessor"], obs)
            obs = preprocessor_output.output
            network_state["preprocessor"] = preprocessor_output.next_state
            regularization_loss += preprocessor_output.regularization_loss
        actor_output = self.actor(network_state["actor"], obs)
        sampler_output = self.action_sampler(
            network_state["action_sampler"], actor_output.output, raw_action
        )
        action, raw_action, loglikelihood = sampler_output.output
        critic_output = self.critic(network_state["critic"], obs)

        network_state["actor"] = actor_output.next_state
        network_state["action_sampler"] = sampler_output.next_state
        network_state["critic"] = critic_output.next_state
        regularization_loss += actor_output.regularization_loss
        regularization_loss += sampler_output.regularization_loss
        regularization_loss += critic_output.regularization_loss
        return network_state, PPONetworkOutput(
            actions=action,
            raw_actions=raw_action,
            loglikelihoods=loglikelihood,
            regularization_loss=regularization_loss,
            value_estimates=critic_output.output,
            metrics={
                "preprocessor": (
                    preprocessor_output.metrics if self.preprocessor is not None else {}
                ),
                "actor": actor_output.metrics,
                "critic": critic_output.metrics,
                "action_sampler": sampler_output.metrics,
            },
        )

    @property
    def components(self):
        components = {
            "actor": self.actor,
            "critic": self.critic,
            "action_sampler": self.action_sampler,
        }
        if self.preprocessor is not None:
            components["preprocessor"] = self.preprocessor
        return components

    def initialize_state(self, batch_size: int) -> ModuleState:
        return {k: v.initialize_state(batch_size) for k, v in self.components.items()}

    def reset_state(self, prev_state: ModuleState) -> ModuleState:
        return {k: v.reset_state(prev_state[k]) for k, v in self.components.items()}

    def update_statistics(
        self, last_rollout: Transition, total_steps: jax.Array
    ) -> None:
        for comp in self.components.values():
            comp.update_statistics(last_rollout, total_steps)


class Sequential(StatefulModule):
    def __init__(self, layers: List[StatefulModule]):
        self.layers = nnx.List(layers)

    def __call__(self, network_state: List, obs) -> StatefulModuleOutput:
        new_network_state = []
        x = obs
        regularization_loss = jp.array(0.0)
        metrics = {}
        for layer, layer_state in zip(self.layers, network_state):
            layer_output = layer(layer_state, x)
            new_state = layer_output.next_state
            x = layer_output.output
            new_network_state.append(new_state)
            regularization_loss += layer_output.regularization_loss
            metrics[len(metrics)] = layer_output.metrics
        return StatefulModuleOutput(new_network_state, x, regularization_loss, metrics)

    def initialize_state(self, batch_size) -> List:
        state = []
        for layer in self.layers:
            state.append(layer.initialize_state(batch_size))
        return state

    def reset_state(self, prev_state) -> List:
        new_states = []
        for layer, layer_prev_state in zip(self.layers, prev_state):
            new_states.append(layer.reset_state(layer_prev_state))
        return new_states

    def __getitem__(self, ind) -> StatefulModule:
        return self.layers[ind]

    def update_statistics(
        self, last_rollout: Transition, total_steps: jax.Array
    ) -> None:
        for layer in self.layers:
            layer.update_statistics(last_rollout, total_steps)


class Concat(StatefulModule):
    def __init__(self, **kwargs):
        self.components = nnx.Dict(kwargs)

    def __call__(self, state, x) -> StatefulModuleOutput:
        new_state = {}
        regularization_loss = jp.array(0.0)
        outputs = []
        metrics = {}
        for key, component in self.components.items():
            component_input = x[key]
            component_output = component(state[key], component_input)
            regularization_loss += component_output.regularization_loss
            new_state[key] = component_output.next_state
            metrics[key] = component_output.metrics
            outputs.append(component_output.output)
        concated = jp.concatenate(outputs, axis=-1)
        return StatefulModuleOutput(new_state, concated, regularization_loss, metrics)

    def initialize_state(self, batch_size: int) -> Any:
        return {k: c.initialize_state(batch_size) for k, c in self.components.items()}

    def reset_state(self, prev_state) -> Any:
        return {k: c.reset_state(prev_state[k]) for k, c in self.components.items()}

    def update_statistics(
        self, last_rollout: Transition, total_steps: jax.Array
    ) -> None:
        for component in self.components.values():
            component.update_statistics(last_rollout, total_steps)


class Flattener(StatefulModule):
    """Takes a PyTree as input and concatenates each leaf along the last axis."""

    def __call__(self, state, x):
        flattened, _ = jax.tree.flatten(x)
        flattened = [a.reshape((a.shape[0], -1)) for a in flattened]
        concated = jp.concatenate(flattened, axis=-1)
        return StatefulModuleOutput((), concated, jp.array(0.0), {})
