"""Small utility :class:`StatefulModule` s.

These are stateless layers that handle pytree projection / reshaping /
scalar arithmetic. They thread ``context`` through but don't read it.

Available:

* :class:`Flattener` — flatten a pytree into one tensor (depth 0), or
  preserve the top ``preserve_levels`` levels of dict/list/tuple
  structure and flatten below.
* :class:`Filter` — declarative pytree extraction/projection. Takes a
  dict spec keyed by output name; each entry is a string (top-level
  key), a tuple of strings/ints (nested path), or a callable applied
  to the full input.
* :class:`Scale` — multiply by a fixed scalar.
* :class:`Merge` — run several named sub-modules on the same input,
  each producing a dict, and merge them into one flat dict. The
  natural complement to :class:`Parallel` when downstream consumers
  (e.g. :class:`~nnx_ppo.algorithms.adapter.PPOAdapter`) want one
  flat dict of named heads.
"""

from collections.abc import Callable
from typing import Any, Union

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import (
    Context,
    ModuleState,
    StatefulModule,
    StatefulModuleOutput,
)


FilterSpec = Union[str, tuple, Callable[[Any], Any]]


def _resolve_components(
    name: str,
    modules: dict | None,
    kwargs: dict,
) -> dict:
    """Helper for containers that accept either ``X({k: v, ...})`` or
    ``X(k=v, ...)`` construction.

    Returns the chosen component dict; raises if both forms are used or
    neither is non-empty.
    """
    if modules is not None and kwargs:
        raise ValueError(
            f"{name}: pass either a positional dict or keyword arguments, "
            "not both"
        )
    components = modules if modules is not None else kwargs
    if not components:
        raise ValueError(f"{name} requires at least one component")
    return components


class Flattener(StatefulModule):
    """Flatten a pytree into a tensor (or a dict-of-tensors).

    With the default ``preserve_levels=0`` every leaf is reshaped to
    ``(B, -1)`` and concatenated along the last axis, producing one
    flat tensor.

    With ``preserve_levels=N``, the top ``N`` levels of ``dict`` /
    ``list`` / ``tuple`` structure are preserved and only the
    sub-trees below them are flattened. So ``Flattener(preserve_levels=1)``
    applied to ``{"a": {"p": (B, 4), "t": (B, 8)}, "b": (B, 6)}``
    returns ``{"a": (B, 12), "b": (B, 6)}``.

    Idempotent on already-flat inputs at the appropriate depth: passing
    ``{"a": (B, 12), "b": (B, 6)}`` through ``Flattener(preserve_levels=1)``
    is a no-op (each value is reshape-and-concat of a single leaf).
    """

    def __init__(self, preserve_levels: int = 0):
        if preserve_levels < 0:
            raise ValueError(
                f"preserve_levels must be >= 0, got {preserve_levels}"
            )
        self.preserve_levels = preserve_levels

    def __call__(
        self,
        state: tuple[()],
        x: Any,
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        output = _flatten_at_depth(x, self.preserve_levels)
        return StatefulModuleOutput((), output, jp.array(0.0), {})


def _flatten_at_depth(x: Any, preserve_levels: int) -> Any:
    if preserve_levels == 0:
        leaves, _ = jax.tree.flatten(x)
        return jp.concatenate(
            [a.reshape((a.shape[0], -1)) for a in leaves], axis=-1
        )
    if isinstance(x, dict):
        return {k: _flatten_at_depth(v, preserve_levels - 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(
            _flatten_at_depth(v, preserve_levels - 1) for v in x
        )
    raise TypeError(
        "Flattener(preserve_levels > 0) requires dict/list/tuple at each "
        f"preserved level; encountered a leaf of type {type(x).__name__} "
        f"with {preserve_levels} levels still to preserve."
    )


class Filter(StatefulModule):
    """Declarative pytree extraction / projection.

    ``spec`` is a dict ``{output_key: extraction}`` where each extraction
    is one of:

    * a **string** ``k`` — take ``x[k]``;
    * a **tuple** of strings/ints ``(k1, k2, ...)`` — nested path,
      equivalent to ``x[k1][k2]...``;
    * a **callable** ``fn`` — applied to the full input ``x``; ``fn(x)``
      becomes the value for ``output_key``.

    The result is a dict with the same keys as ``spec``. Anything in
    the input not named (directly or via a callable) is dropped.

    Use it to ablate observation streams (mask out goal information
    from the actor), to give the critic privileged info, or to reshape
    structured observations before a downstream consumer that expects
    a flat dict.

    Example::

        actor_filter = Filter({
            "arm_L":  ("arm_L", "proprioception"),
            "arm_R":  ("arm_R", "proprioception"),
            "root":   lambda obs: jp.zeros_like(obs["root"]["pos"][..., :0]),
        })
    """

    def __init__(self, spec: dict[str, FilterSpec]):
        if not isinstance(spec, dict):
            raise TypeError(
                f"Filter spec must be a dict; got {type(spec).__name__}"
            )
        for out_key, sub in spec.items():
            if not isinstance(sub, (str, tuple)) and not callable(sub):
                raise TypeError(
                    f"Filter spec for {out_key!r} must be str, tuple, or "
                    f"callable; got {type(sub).__name__}"
                )
        self._spec = dict(spec)

    def __call__(
        self,
        state: tuple[()],
        x: Any,
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        output: dict[str, Any] = {}
        for out_key, sub in self._spec.items():
            if isinstance(sub, str):
                output[out_key] = x[sub]
            elif isinstance(sub, tuple):
                v = x
                for p in sub:
                    v = v[p]
                output[out_key] = v
            else:  # callable
                output[out_key] = sub(x)
        return StatefulModuleOutput((), output, jp.array(0.0), {})


class Scale(StatefulModule):
    """Multiply the input by a fixed scalar factor.

    Cheaper to read than burying the factor inside a Dense's
    initializer, and keeps the factor visible to downstream
    introspection (logging, weight inspection).
    """

    def __init__(self, factor: float):
        self.factor = float(factor)

    def __call__(
        self,
        state: tuple[()],
        x: Any,
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        return StatefulModuleOutput(
            state, jax.tree.map(lambda v: v * self.factor, x), jp.array(0.0), {}
        )


class Merge(StatefulModule):
    """Run named sub-modules on the same input; merge their dict outputs.

    Each sub-module must return a ``dict``. The outputs are merged into
    one flat dict; duplicate keys across components are a hard error.
    Use this when a downstream consumer (e.g.
    :class:`~nnx_ppo.algorithms.adapter.PPOAdapter`) wants one flat
    dict of named heads but the heads come from independent
    sub-networks::

        inner = Merge(
            motors=graph,                    # emits {motor_arm_L, ...}
            critic=detached_critic_stack,    # emits {value_arm_L, ...}
        )

    Accepts either keyword arguments (when names are valid Python
    identifiers) or a positional dict ``Merge({...})`` (when they are
    not).

    Carry state is a dict ``{name: component_state}`` — same shape as
    :class:`~nnx_ppo.networks.containers.Parallel`.
    """

    def __init__(
        self,
        modules: dict[str, StatefulModule] | None = None,
        /,
        **kwargs: StatefulModule,
    ):
        components = _resolve_components("Merge", modules, kwargs)
        self.components = nnx.Dict(components)

    def __call__(
        self,
        state: dict[str, ModuleState],
        x: Any,
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        new_state: dict[str, ModuleState] = {}
        merged: dict[str, Any] = {}
        regularization_loss = jp.array(0.0)
        metrics: dict[str, Any] = {}
        for name, component in self.components.items():
            out = component(state[name], x, context=context)
            new_state[name] = out.next_state
            regularization_loss += out.regularization_loss
            metrics[name] = out.metrics
            if not isinstance(out.output, dict):
                raise TypeError(
                    f"Merge component {name!r} must return a dict; got "
                    f"{type(out.output).__name__}"
                )
            for k, v in out.output.items():
                if k in merged:
                    raise ValueError(
                        f"Merge: duplicate key {k!r} produced by multiple "
                        f"components"
                    )
                merged[k] = v
        return StatefulModuleOutput(
            new_state, merged, regularization_loss, metrics
        )

    def initialize_state(self, batch_size: int) -> dict[str, ModuleState]:
        return {
            k: c.initialize_state(batch_size) for k, c in self.components.items()
        }

    def reset_state(
        self, prev_state: dict[str, ModuleState]
    ) -> dict[str, ModuleState]:
        return {
            k: c.reset_state(prev_state[k]) for k, c in self.components.items()
        }


class Map(StatefulModule):
    """Per-key dispatch: dict input, dict output.

    Each named sub-module sees the upstream's same-named entry as input
    and produces the same-named entry of the output. Distinct from:

    * :class:`~nnx_ppo.networks.containers.Parallel` — same input fed to
      every component, dict output.
    * :class:`~nnx_ppo.networks.containers.Concat` — per-key dispatch,
      concatenated output (no dict).

    Use this when you want to apply a different module to each entry of
    a structured input — e.g. a per-population action sampler::

        Map({pop: NormalTanhSampler(rngs, ...) for pop in POPULATIONS})

    The input dict must contain at least every key in ``modules``; extra
    keys are dropped.

    Accepts either keyword arguments or a positional dict
    ``Map({...})`` (necessary when keys are not valid Python
    identifiers).

    Carry state is a dict ``{name: component_state}``.
    """

    def __init__(
        self,
        modules: dict[str, StatefulModule] | None = None,
        /,
        **kwargs: StatefulModule,
    ):
        components = _resolve_components("Map", modules, kwargs)
        self.components = nnx.Dict(components)

    def __call__(
        self,
        state: dict[str, ModuleState],
        x: dict[str, Any],
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        new_state: dict[str, ModuleState] = {}
        outputs: dict[str, Any] = {}
        regularization_loss = jp.array(0.0)
        metrics: dict[str, Any] = {}
        for name, component in self.components.items():
            out = component(state[name], x[name], context=context)
            new_state[name] = out.next_state
            outputs[name] = out.output
            regularization_loss = regularization_loss + out.regularization_loss
            metrics[name] = out.metrics
        return StatefulModuleOutput(
            new_state, outputs, regularization_loss, metrics
        )

    def initialize_state(self, batch_size: int) -> dict[str, ModuleState]:
        return {
            k: c.initialize_state(batch_size) for k, c in self.components.items()
        }

    def reset_state(
        self, prev_state: dict[str, ModuleState]
    ) -> dict[str, ModuleState]:
        return {
            k: c.reset_state(prev_state[k]) for k, c in self.components.items()
        }
