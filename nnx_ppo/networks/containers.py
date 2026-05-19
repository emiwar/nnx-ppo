from typing import Any, Optional
from collections.abc import Sequence
from warnings import deprecated

import jax
import jax.numpy as jp
from flax import nnx
from jaxtyping import Array, Float, PyTree, ScalarLike

from nnx_ppo.algorithms.types import Transition
from nnx_ppo.networks.sampling_layers import ActionSampler
from nnx_ppo.networks.types import (
    Context,
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
        self,
        network_state: dict[str, ModuleState],
        obs: PyTree[Float[Array, "..."]],
        raw_action: Optional[Float[Array, "batch action_dim"]] = None,
        *,
        context: Context = Context.INFERENCE,
    ) -> tuple[dict[str, ModuleState], PPONetworkOutput]:
        regularization_loss = jp.array(0.0)
        preprocessor_metrics: dict = {}
        if self.preprocessor is not None:
            preprocessor_output = self.preprocessor(
                network_state["preprocessor"], obs, context=context
            )
            obs = preprocessor_output.output
            network_state["preprocessor"] = preprocessor_output.next_state
            regularization_loss += preprocessor_output.regularization_loss
            preprocessor_metrics = preprocessor_output.metrics
        actor_output = self.actor(network_state["actor"], obs, context=context)
        sampler_output = self.action_sampler(
            network_state["action_sampler"],
            actor_output.output,
            raw_action,
            context=context,
        )
        action, raw_action, loglikelihood = sampler_output.output
        critic_output = self.critic(network_state["critic"], obs, context=context)

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
            value_estimates=jp.squeeze(critic_output.output, axis=-1),
            metrics={
                "preprocessor": preprocessor_metrics,
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

    def initialize_state(self, batch_size: int) -> dict[str, ModuleState]:
        return {k: v.initialize_state(batch_size) for k, v in self.components.items()}

    def reset_state(self, prev_state: dict[str, ModuleState]) -> dict[str, ModuleState]:
        return {k: v.reset_state(prev_state[k]) for k, v in self.components.items()}

    def update_statistics(
        self, last_rollout: Transition, total_steps: ScalarLike
    ) -> None:
        for comp in self.components.values():
            comp.update_statistics(last_rollout, total_steps)


class Sequential(StatefulModule):
    def __init__(self, layers: Sequence[StatefulModule]):
        self.layers = nnx.List(layers)

    def __call__(
        self,
        network_state: list[ModuleState],
        obs: Any,
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        new_network_state = []
        x = obs
        regularization_loss = jp.array(0.0)
        metrics = {}
        for layer, layer_state in zip(self.layers, network_state):
            layer_output = layer(layer_state, x, context=context)
            new_state = layer_output.next_state
            x = layer_output.output
            new_network_state.append(new_state)
            regularization_loss += layer_output.regularization_loss
            metrics[len(metrics)] = layer_output.metrics
        return StatefulModuleOutput(new_network_state, x, regularization_loss, metrics)

    def initialize_state(self, batch_size: int) -> list[ModuleState]:
        state = []
        for layer in self.layers:
            state.append(layer.initialize_state(batch_size))
        return state

    def reset_state(self, prev_state: list[ModuleState]) -> list[ModuleState]:
        new_states = []
        for layer, layer_prev_state in zip(self.layers, prev_state):
            new_states.append(layer.reset_state(layer_prev_state))
        return new_states

    def __getitem__(self, ind: int) -> StatefulModule:
        return self.layers[ind]

    def update_statistics(
        self, last_rollout: Transition, total_steps: ScalarLike
    ) -> None:
        for layer in self.layers:
            layer.update_statistics(last_rollout, total_steps)


class Concat(StatefulModule):
    def __init__(self, **kwargs: StatefulModule):
        self.components = nnx.Dict(kwargs)

    def __call__(
        self,
        state: dict[str, ModuleState],
        x: dict[str, Any],
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        new_state = {}
        regularization_loss = jp.array(0.0)
        outputs = []
        metrics = {}
        for key, component in self.components.items():
            component_input = x[key]
            component_output = component(state[key], component_input, context=context)
            regularization_loss += component_output.regularization_loss
            new_state[key] = component_output.next_state
            metrics[key] = component_output.metrics
            outputs.append(component_output.output)
        concated = jp.concatenate(outputs, axis=-1)
        return StatefulModuleOutput(new_state, concated, regularization_loss, metrics)

    def initialize_state(self, batch_size: int) -> dict[str, ModuleState]:
        return {k: c.initialize_state(batch_size) for k, c in self.components.items()}

    def reset_state(self, prev_state: dict[str, ModuleState]) -> dict[str, ModuleState]:
        return {k: c.reset_state(prev_state[k]) for k, c in self.components.items()}

    def update_statistics(
        self, last_rollout: Transition, total_steps: ScalarLike
    ) -> None:
        for component in self.components.values():
            component.update_statistics(last_rollout, total_steps)


class Flattener(StatefulModule):
    """Takes a PyTree as input and concatenates each leaf along the last axis."""

    def __call__(
        self,
        state: tuple[()],
        x: Any,
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        flattened, _ = jax.tree.flatten(x)
        flattened = [a.reshape((a.shape[0], -1)) for a in flattened]
        concated = jp.concatenate(flattened, axis=-1)
        return StatefulModuleOutput((), concated, jp.array(0.0), {})


class Parallel(StatefulModule):
    """Runs several sub-modules on the **same** input and returns their outputs
    as a dict keyed by sub-module name.

    Typical use: assemble a trunk that produces both action-distribution
    parameters and value estimates from shared upstream features::

        trunk = Sequential([
            shared_encoder,
            Parallel(action_params=actor_head, value=critic_head),
        ])
        # trunk(state, x).output is {"action_params": ..., "value": ...}
    """

    def __init__(self, **components: StatefulModule):
        if not components:
            raise ValueError("Parallel requires at least one sub-module")
        self.components = nnx.Dict(components)

    def __call__(
        self,
        state: dict[str, ModuleState],
        x: Any,
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        new_state: dict[str, ModuleState] = {}
        outputs: dict[str, Any] = {}
        regularization_loss = jp.array(0.0)
        metrics: dict[str, Any] = {}
        for key, component in self.components.items():
            out = component(state[key], x, context=context)
            new_state[key] = out.next_state
            outputs[key] = out.output
            regularization_loss += out.regularization_loss
            metrics[key] = out.metrics
        return StatefulModuleOutput(new_state, outputs, regularization_loss, metrics)

    def initialize_state(self, batch_size: int) -> dict[str, ModuleState]:
        return {k: c.initialize_state(batch_size) for k, c in self.components.items()}

    def reset_state(self, prev_state: dict[str, ModuleState]) -> dict[str, ModuleState]:
        return {k: c.reset_state(prev_state[k]) for k, c in self.components.items()}

    def update_statistics(
        self, last_rollout: Transition, total_steps: ScalarLike
    ) -> None:
        for component in self.components.values():
            component.update_statistics(last_rollout, total_steps)


class Splitter(StatefulModule):
    """Splits a single input tensor into a dict of named slices along the last axis.

    Used at the end of a stack to turn a flat tensor head into a structured
    dict output that an adapter can route to samplers / value specs::

        Sequential([
            trunk,
            Dense(hidden, 2 * action_size + 1, rngs),
            Splitter(action_params=2 * action_size, value=1),
        ])

    With a single keyword (``Splitter(action_params=N)``) the layer simply
    relabels the input as a dict, taking the first N features.

    The slices are taken in keyword-argument insertion order. The sum of
    declared sizes must not exceed the input's last-axis size; any excess
    input features are silently ignored, matching plain slicing semantics.
    """

    def __init__(self, **sizes: int):
        if not sizes:
            raise ValueError("Splitter requires at least one named slice")
        for k, v in sizes.items():
            if v <= 0:
                raise ValueError(f"slice size for {k!r} must be positive, got {v}")
        self._sizes = dict(sizes)

    def __call__(
        self,
        state: tuple[()],
        x: Any,
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        outputs: dict[str, Any] = {}
        offset = 0
        for key, size in self._sizes.items():
            outputs[key] = x[..., offset : offset + size]
            offset += size
        return StatefulModuleOutput((), outputs, jp.array(0.0), {})
