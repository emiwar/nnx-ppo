from typing import Any
from collections.abc import Sequence

import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import (
    StatefulModule,
    ModuleState,
    StatefulModuleOutput,
)


class Sequential(StatefulModule):
    def __init__(self, layers: Sequence[StatefulModule]):
        self.layers = nnx.List(layers)

    def __call__(
        self,
        network_state: list[ModuleState],
        obs: Any,
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        new_network_state = []
        new_extras: list[Any] = []
        x = obs
        regularization_loss = jp.array(0.0)
        metrics = {}
        for i, (layer, layer_state) in enumerate(zip(self.layers, network_state)):
            layer_extras = None if rollout_extras is None else rollout_extras[i]
            layer_output = layer(layer_state, x, layer_extras)
            new_network_state.append(layer_output.next_state)
            new_extras.append(layer_output.rollout_extras)
            x = layer_output.output
            regularization_loss += layer_output.regularization_loss
            metrics[len(metrics)] = layer_output.metrics
        return StatefulModuleOutput(
            new_network_state, x, regularization_loss, metrics, new_extras
        )

    def initialize_state(self, batch_size: int) -> list[ModuleState]:
        return [layer.initialize_state(batch_size) for layer in self.layers]

    def reset_state(self, prev_state: list[ModuleState]) -> list[ModuleState]:
        return [layer.reset_state(s) for layer, s in zip(self.layers, prev_state)]

    def update_statistics(self, rollout_extras: Any) -> None:
        for layer, layer_extras in zip(self.layers, rollout_extras):
            layer.update_statistics(layer_extras)

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
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        new_state: dict[str, ModuleState] = {}
        new_extras: dict[str, Any] = {}
        regularization_loss = jp.array(0.0)
        outputs = []
        metrics: dict[str, Any] = {}
        for key, component in self.components.items():
            child_extras = None if rollout_extras is None else rollout_extras[key]
            out = component(state[key], x[key], child_extras)
            regularization_loss += out.regularization_loss
            new_state[key] = out.next_state
            new_extras[key] = out.rollout_extras
            metrics[key] = out.metrics
            outputs.append(out.output)
        concated = jp.concatenate(outputs, axis=-1)
        return StatefulModuleOutput(
            new_state, concated, regularization_loss, metrics, new_extras
        )

    def initialize_state(self, batch_size: int) -> dict[str, ModuleState]:
        return {k: c.initialize_state(batch_size) for k, c in self.components.items()}

    def reset_state(self, prev_state: dict[str, ModuleState]) -> dict[str, ModuleState]:
        return {k: c.reset_state(prev_state[k]) for k, c in self.components.items()}

    def update_statistics(self, rollout_extras: Any) -> None:
        for key, component in self.components.items():
            component.update_statistics(rollout_extras[key])


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
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        new_state: dict[str, ModuleState] = {}
        new_extras: dict[str, Any] = {}
        outputs: dict[str, Any] = {}
        regularization_loss = jp.array(0.0)
        metrics: dict[str, Any] = {}
        for key, component in self.components.items():
            child_extras = None if rollout_extras is None else rollout_extras[key]
            out = component(state[key], x, child_extras)
            new_state[key] = out.next_state
            new_extras[key] = out.rollout_extras
            outputs[key] = out.output
            regularization_loss += out.regularization_loss
            metrics[key] = out.metrics
        return StatefulModuleOutput(
            new_state, outputs, regularization_loss, metrics, new_extras
        )

    def initialize_state(self, batch_size: int) -> dict[str, ModuleState]:
        return {k: c.initialize_state(batch_size) for k, c in self.components.items()}

    def reset_state(self, prev_state: dict[str, ModuleState]) -> dict[str, ModuleState]:
        return {k: c.reset_state(prev_state[k]) for k, c in self.components.items()}

    def update_statistics(self, rollout_extras: Any) -> None:
        for key, component in self.components.items():
            component.update_statistics(rollout_extras[key])


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
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        outputs: dict[str, Any] = {}
        offset = 0
        for key, size in self._sizes.items():
            outputs[key] = x[..., offset : offset + size]
            offset += size
        return StatefulModuleOutput((), outputs, jp.array(0.0), {}, None)
