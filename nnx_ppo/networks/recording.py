"""Activation recording for network modules.

Recording is an *eval/analysis-only* feature for inspecting the per-unit
activations of a network. The design keeps all recording logic out of the
modules themselves: a leaf module's forward ``output`` already *is* its
activation, and the containers already propagate each child's ``metrics`` up
the tree into a nested, named dict. We exploit both.

:func:`with_recording` returns a *separate* copy of a network in which every
leaf :class:`~nnx_ppo.networks.types.StatefulModule` is wrapped in a
:class:`Recorder`. A ``Recorder`` runs its wrapped module unchanged and injects
that module's ``output`` into the returned ``StatefulModuleOutput.metrics`` under
the reserved key :data:`ACTIVATION_KEY`. The activation therefore rides the
existing ``metrics`` channel to the top-level call for free, where
:func:`extract_activations` pulls it back out, keyed by each module's structural
path in the container tree (e.g. ``action/0``, ``action/1``, ``value/...``).

Because the original network is never modified, recording is "off" by absence:
there is no ``if record:`` branch anywhere, and the unwrapped network behaves
identically jitted, non-jitted, and in tests.

The graph network (:class:`~nnx_ppo.networks.graph.graph.PopulationGraph`) is a
special case: its interesting units are internal *populations* (plain dataclasses,
not sub-modules), and it drops its children's metrics. Instead of being
leaf-wrapped, it has its own ``record_activations`` flag that
:func:`with_recording` enables; when set it emits all population activations into
its metrics under :data:`ACTIVATION_KEY`.

Per-timestep activations can only leave an ``nnx.scan`` via its ``ys`` return
(stacking, not reduction). ``eval_rollout`` *reduces* ``metrics`` to scalars and
is therefore not usable for per-unit recording. Use a plain Python eval loop
(calling :func:`extract_activations` on ``out.metrics`` each step) or the
convenience :func:`~nnx_ppo.algorithms.rollout.record_activations_rollout`.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from typing import Any

from flax import nnx

from nnx_ppo.networks.graph.graph import PopulationGraph
from nnx_ppo.networks.types import (
    ACTIVATION_KEY,
    ModuleState,
    StatefulModule,
    StatefulModuleOutput,
)

# ``ACTIVATION_KEY`` is defined in ``networks.types`` (so modules with custom
# recording can use it without importing this module, which would be a circular
# import) and re-exported here as part of the recording API.
__all__ = ["ACTIVATION_KEY", "Recorder", "with_recording", "extract_activations"]


class Recorder(StatefulModule):
    """Wraps a single leaf module and records its forward ``output``.

    Delegates every part of the :class:`StatefulModule` interface to the wrapped
    module verbatim; the only addition is the wrapped module's ``output`` placed
    into the returned ``metrics`` under :data:`ACTIVATION_KEY`. The forward
    ``output`` / ``next_state`` / ``rollout_extras`` pass through unchanged, so a
    recording network's forward pass is bit-identical to the original.
    """

    def __init__(self, wrapped: StatefulModule):
        self.wrapped = wrapped

    def __call__(
        self,
        module_state: ModuleState,
        obs: Any,
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        out = self.wrapped(module_state, obs, rollout_extras)
        return dataclasses.replace(
            out, metrics={**out.metrics, ACTIVATION_KEY: out.output}
        )

    def initialize_state(self, batch_size: int) -> ModuleState:
        return self.wrapped.initialize_state(batch_size)

    def reset_state(self, prev_state: ModuleState) -> ModuleState:
        return self.wrapped.reset_state(prev_state)

    def update_statistics(self, rollout_extras: Any) -> None:
        return self.wrapped.update_statistics(rollout_extras)


def _is_leaf(module: Any) -> bool:
    """True for a ``StatefulModule`` with no ``StatefulModule`` descendants.

    Container modules hold their children inside ``nnx.List`` / ``nnx.Dict``
    (themselves ``nnx.Module``\\ s but not ``StatefulModule``\\ s), so we look at
    all descendants rather than only immediate children. ``nnx.iter_modules``
    yields the module itself with an empty path; descendants have a non-empty
    path.
    """
    if not isinstance(module, StatefulModule):
        return False
    for path, sub in nnx.iter_modules(module):
        if path and isinstance(sub, StatefulModule):
            return False
    return True


def with_recording(net: StatefulModule) -> StatefulModule:
    """Return a recording copy of ``net``; the original is left untouched.

    Every leaf module is wrapped in a :class:`Recorder`; every
    :class:`PopulationGraph` has its ``record_activations`` flag enabled. The
    rebuild uses ``nnx.recursive_map`` (bottom-up), so leaves are wrapped exactly
    once and parameters are shared with the original (read-only at eval).

    If ``net`` already contains :class:`Recorder` layers it is already
    recordable; a warning is issued and it is returned unchanged (wrapping it
    again would double-wrap the inner leaves).
    """
    if any(isinstance(m, Recorder) for _, m in nnx.iter_modules(net)):
        warnings.warn(
            "with_recording() was called on a network that is already "
            "recordable (it contains Recorder layers); returning it unchanged.",
            stacklevel=2,
        )
        return net

    def _wrap(path: tuple[Any, ...], node: Any) -> Any:
        # Graph is handled before the leaf check: a connection-less graph would
        # otherwise look like a leaf, and we want the flag, not a Recorder.
        if isinstance(node, PopulationGraph):
            node.record_activations = True
            return node
        if _is_leaf(node):
            return Recorder(node)
        return node

    return nnx.recursive_map(_wrap, net)


def extract_activations(metrics: Any) -> Any:
    """Pull the recorded activations out of a network's ``out.metrics`` tree.

    Walks the nested metrics dict and returns a tree of the same container shape
    in which each recorded module is replaced by its activation (the value stored
    under :data:`ACTIVATION_KEY`). Real scalar metrics and branches without any
    activations are dropped. Returns ``None`` if there are no activations (e.g.
    the network was not made recordable).
    """
    if not isinstance(metrics, Mapping):
        return None
    if ACTIVATION_KEY in metrics:
        # A recorded module: return its activation, ignoring sibling metrics.
        return metrics[ACTIVATION_KEY]
    out: dict[Any, Any] = {}
    for key, value in metrics.items():
        sub = extract_activations(value)
        if sub is not None:
            out[key] = sub
    return out if out else None
