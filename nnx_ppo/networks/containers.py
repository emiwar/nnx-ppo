from typing import Any, Optional
from collections.abc import Sequence

import jax.numpy as jp
from flax import nnx

from nnx_ppo.algorithms.adapter import PPOAdapter
from nnx_ppo.algorithms.distributions import ActionSampler
from nnx_ppo.networks.types import (
    Context,
    StatefulModule,
    ModuleState,
    StatefulModuleOutput,
)


class PPOActorCritic(PPOAdapter):
    """Convenience PPO network for the standard one-actor / one-critic case.

    A thin subclass of :class:`~nnx_ppo.algorithms.adapter.PPOAdapter` that
    wires up the conventional decomposition: one actor producing action
    distribution parameters, one critic producing a scalar value estimate,
    and an optional preprocessor (typically a :class:`Normalizer`) sitting
    in front of both.

    Use this for the typical single-agent setup. Drop down to bare
    ``PPOAdapter`` when you need modular or multi-head networks (e.g. one
    sampler per body-module, per-population value heads).

    The ``actor``, ``critic``, ``action_sampler``, and ``preprocessor``
    constructor arguments are also exposed as attributes after init, so
    external code can introspect them (e.g. for parameter logging).
    """

    def __init__(
        self,
        actor: StatefulModule,
        critic: StatefulModule,
        action_sampler: ActionSampler,
        preprocessor: Optional[StatefulModule] = None,
    ):
        # Expose direct references for backwards compatibility with code
        # that introspects networks.actor / .critic / .action_sampler /
        # .preprocessor (factories_test, metrics.py, checkpointing_test,
        # vnl-experiments). NNX handles shared references to the same
        # module instance across attributes.
        self.actor = actor
        self.critic = critic
        self.action_sampler = action_sampler
        self.preprocessor = preprocessor

        inner: StatefulModule = Parallel(action_params=actor, value=critic)
        if preprocessor is not None:
            inner = Sequential([preprocessor, inner])
        super().__init__(
            inner=inner,
            action_specs={"action_params": action_sampler},
            value_specs="value",
        )


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


class Concat(StatefulModule):
    """Per-key dispatch + concat: dict input, single-tensor output.

    Each named sub-module sees the upstream's same-named entry as
    input; the per-component outputs are concatenated along the last
    axis. Accepts either keyword arguments or a positional dict
    ``Concat({...})`` (when keys are not valid Python identifiers).
    """

    def __init__(
        self,
        modules: dict[str, StatefulModule] | None = None,
        /,
        **kwargs: StatefulModule,
    ):
        if modules is not None and kwargs:
            raise ValueError(
                "Concat: pass either a positional dict or keyword "
                "arguments, not both"
            )
        components = modules if modules is not None else kwargs
        if not components:
            raise ValueError("Concat requires at least one component")
        self.components = nnx.Dict(components)

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

    def __init__(
        self,
        modules: dict[str, StatefulModule] | None = None,
        /,
        **kwargs: StatefulModule,
    ):
        if modules is not None and kwargs:
            raise ValueError(
                "Parallel: pass either a positional dict or keyword "
                "arguments, not both"
            )
        components = modules if modules is not None else kwargs
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
