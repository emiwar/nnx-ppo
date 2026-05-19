"""Population-graph container.

:class:`PopulationGraph` is a :class:`StatefulModule` that owns a set of
named :class:`Population` nodes and typed :class:`Connection` edges
between them. It exposes a small declarative API
(``add_population`` / ``connect`` / ``add_output`` / ``finalize``) and
handles the per-step orchestration:

* topological sort over delay-0 edges (delay-0 cycles are a hard error);
* per-population sum integration of incoming connections;
* per-population activation (transfer function) applied **once** after
  integration — connections are linear by default;
* one shared circular output buffer per population, sized by the
  maximum outgoing delay, storing post-activation outputs that are read
  by all outgoing delayed connections;
* declarative ``add_output(name, source=..., head=...)`` mapping
  population activations through optional heads into a named output
  dict, ready to feed an adapter such as :class:`PPOAdapter`.

Use :class:`PopulationGraph` when you have a graph with multiple
connections per node, recurrent loops, or per-connection delays.
For straight encoder/decoder stacks ``Sequential`` is simpler.
"""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.graph.connection import Connection
from nnx_ppo.networks.graph.population import Population
from nnx_ppo.networks.types import (
    Context,
    ModuleState,
    StatefulModule,
    StatefulModuleOutput,
)


class _Identity(StatefulModule):
    """Stateless pass-through used as the default population compute."""

    def __call__(
        self,
        state: tuple[()],
        x: Any,
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        return StatefulModuleOutput(state, x, jp.array(0.0), {})


class PopulationGraph(StatefulModule):
    """Declarative population/connection graph as a :class:`StatefulModule`.

    Construction is two-phase:

    1. ``add_population`` / ``connect`` / ``add_output`` to describe the
       graph.
    2. ``finalize()`` to validate (cycle detection, shape inference,
       buffer sizing) and freeze the graph for forward passes.

    After ``finalize()``, the graph behaves as any other
    :class:`StatefulModule`: it can be a layer in ``Sequential``, a
    population's ``compute``, or a connection's ``transform``.
    """

    def __init__(self, rngs: nnx.Rngs):
        self.rngs = rngs
        # Pre-finalize registries (plain Python; promoted to nnx containers
        # in finalize()).
        self._pops: dict[str, Population] = {}
        self._conns: list[Connection] = []
        self._outputs: dict[str, tuple[str, Optional[StatefulModule]]] = {}
        self._finalized = False
        # Filled in by finalize().
        self._topo_order: tuple[str, ...] = ()
        self._incoming: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------------
    # Build API
    # ------------------------------------------------------------------

    def add_population(
        self,
        name: str,
        output_size: int,
        *,
        input_size: Optional[int] = None,
        compute: Optional[StatefulModule] = None,
        activation: Optional[Callable] = None,
        input_from: Optional[str] = None,
    ) -> None:
        """Register a population.

        Args:
          name: Unique identifier for this population.
          output_size: Size of the (post-activation) output vector.
          input_size: Size of the integrated-input vector consumed by
            ``compute``. Defaults to ``output_size``.
          compute: A :class:`StatefulModule` mapping
            ``[B, input_size] -> [B, output_size]``. Defaults to
            :class:`_Identity` (requires ``input_size == output_size``).
          activation: Optional transfer function applied elementwise to
            ``compute``'s output. Connection outputs are summed
            *before* this; activation is applied *once per population
            step*, not per connection.
          input_from: Optional key in the ``obs`` dict whose value is
            added to this population's integrated input. Lets input
            populations read directly from the environment observation.
            When set, ``obs[input_from]`` must have shape
            ``[B, input_size]``.
        """
        self._assert_not_finalized()
        if name in self._pops:
            raise ValueError(f"population {name!r} already exists")
        if input_size is None:
            input_size = output_size
        if compute is None:
            if input_size != output_size:
                raise ValueError(
                    f"population {name!r}: compute=None requires "
                    f"input_size == output_size, got "
                    f"{input_size} != {output_size}"
                )
            compute = _Identity()
        self._pops[name] = Population(
            name=name,
            input_size=input_size,
            output_size=output_size,
            compute=compute,
            activation=activation,
            input_from=input_from,
        )

    def connect(
        self,
        src: str,
        dst: str,
        *,
        transform: Optional[StatefulModule] = None,
        delay: int = 0,
    ) -> None:
        """Add a directed connection from ``src`` to ``dst``.

        Args:
          src: Source population name. Must already be registered.
          dst: Destination population name. Must already be registered.
          transform: A :class:`StatefulModule` mapping
            ``[B, src.output_size] -> [B, dst.input_size]``. Defaults to
            a linear :class:`Dense` of the appropriate shape.
          delay: Integer step delay. ``0`` reads the source's output
            from the current step; ``k >= 1`` reads from ``k`` steps
            ago. Before the buffer fills, delayed reads return zeros.
        """
        self._assert_not_finalized()
        if src not in self._pops:
            raise ValueError(f"unknown source population {src!r}")
        if dst not in self._pops:
            raise ValueError(f"unknown destination population {dst!r}")
        if transform is None:
            transform = Dense(
                self._pops[src].output_size,
                self._pops[dst].input_size,
                rngs=self.rngs,
            )
        self._conns.append(Connection(src=src, dst=dst, transform=transform, delay=delay))

    def add_output(
        self,
        name: str,
        *,
        source: str,
        head: Optional[StatefulModule] = None,
    ) -> None:
        """Expose a population's activation (optionally through a head) as
        a named entry in the graph's output dict.

        Args:
          name: Key under which the value will appear in the forward
            pass's output dict.
          source: Population name to read from.
          head: Optional :class:`StatefulModule` applied to the source
            activation before exposing.
        """
        self._assert_not_finalized()
        if source not in self._pops:
            raise ValueError(f"unknown source population {source!r}")
        if name in self._outputs:
            raise ValueError(f"output {name!r} already declared")
        self._outputs[name] = (source, head)

    def finalize(self) -> None:
        """Validate and freeze the graph.

        Performs:
          * delay-0 cycle detection (any cycle is a hard error);
          * topological ordering of populations (delay-0 edges only);
          * per-population ``max_outgoing_delay`` computation;
          * promotion of populations / connections / heads to NNX
            containers so their parameters are tracked.
        """
        self._assert_not_finalized()

        # max_outgoing_delay per source population.
        for conn in self._conns:
            src_pop = self._pops[conn.src]
            if conn.delay > src_pop.max_outgoing_delay:
                src_pop.max_outgoing_delay = conn.delay

        # Topo sort over delay-0 edges.
        delay0_succ: dict[str, list[str]] = {n: [] for n in self._pops}
        in_degree: dict[str, int] = {n: 0 for n in self._pops}
        for conn in self._conns:
            if conn.delay == 0:
                delay0_succ[conn.src].append(conn.dst)
                in_degree[conn.dst] += 1
        # Kahn's algorithm with deterministic ordering: process in
        # add_population insertion order so finalize() is reproducible.
        ready = [n for n in self._pops if in_degree[n] == 0]
        topo: list[str] = []
        while ready:
            n = ready.pop(0)
            topo.append(n)
            for m in delay0_succ[n]:
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    ready.append(m)
        if len(topo) != len(self._pops):
            unresolved = [n for n in self._pops if n not in topo]
            raise ValueError(
                f"delay-0 cycle detected involving populations: {unresolved}"
            )
        self._topo_order = tuple(topo)

        # Precompute incoming connection indices per destination.
        self._incoming = {n: tuple() for n in self._pops}
        for i, conn in enumerate(self._conns):
            self._incoming[conn.dst] = self._incoming[conn.dst] + (i,)

        # Promote to nnx-tracked containers.
        self.populations = nnx.Dict(self._pops)
        self.connections = nnx.List(self._conns)
        head_dict = {
            name: head
            for name, (_, head) in self._outputs.items()
            if head is not None
        }
        self.output_heads = nnx.Dict(head_dict)
        self._heads_present: frozenset[str] = frozenset(head_dict)
        self._output_sources: dict[str, str] = {
            name: source for name, (source, _) in self._outputs.items()
        }

        self._finalized = True

    def _assert_not_finalized(self) -> None:
        if self._finalized:
            raise RuntimeError("PopulationGraph already finalized")

    def _assert_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError(
                "PopulationGraph must be finalized before use; call finalize()"
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        state: ModuleState,
        obs: Any,
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        self._assert_finalized()
        pop_state = state["populations"]
        conn_state = state["connections"]
        head_state = state["heads"]

        batch_size = jax.tree.leaves(obs)[0].shape[0]
        arange = jp.arange(batch_size)

        new_pop_state: dict[str, dict] = {}
        new_conn_state: list[Any] = [None] * len(self.connections)
        new_head_state: dict[str, Any] = {}
        current_outputs: dict[str, jax.Array] = {}
        reg_loss = jp.array(0.0)
        metrics: dict[str, Any] = {}

        # Process populations in topo order so delay-0 reads see freshly
        # computed source outputs.
        for pop_name in self._topo_order:
            pop = self.populations[pop_name]
            this_pop_state = pop_state[pop_name]

            integrated = jp.zeros((batch_size, pop.input_size))
            if pop.input_from is not None:
                integrated = integrated + obs[pop.input_from]

            for i in self._incoming[pop_name]:
                conn = self.connections[i]
                src_pop = self.populations[conn.src]
                if conn.delay == 0:
                    src_out = current_outputs[conn.src]
                else:
                    L = src_pop.max_outgoing_delay
                    buf = pop_state[conn.src]["buffer"]
                    idx = pop_state[conn.src]["buffer_idx"]
                    read_idx = (idx - conn.delay) % L
                    src_out = buf[arange, read_idx]

                conn_out = conn.transform(
                    conn_state[i], src_out, context=context
                )
                new_conn_state[i] = conn_out.next_state
                integrated = integrated + conn_out.output
                reg_loss = reg_loss + jp.sum(conn_out.regularization_loss)

            compute_out = pop.compute(
                this_pop_state["compute"], integrated, context=context
            )
            reg_loss = reg_loss + jp.sum(compute_out.regularization_loss)
            pre_activation = compute_out.output
            activated = (
                pop.activation(pre_activation)
                if pop.activation is not None
                else pre_activation
            )
            current_outputs[pop_name] = activated
            metrics[pop_name] = compute_out.metrics

            updated = {"compute": compute_out.next_state}
            if pop.max_outgoing_delay > 0:
                buf = this_pop_state["buffer"]
                idx = this_pop_state["buffer_idx"]
                new_buf = buf.at[arange, idx].set(activated)
                new_idx = (idx + 1) % pop.max_outgoing_delay
                updated["buffer"] = new_buf
                updated["buffer_idx"] = new_idx
            new_pop_state[pop_name] = updated

        # Connection transforms not visited above keep their old state.
        for i, s in enumerate(conn_state):
            if new_conn_state[i] is None:
                new_conn_state[i] = s

        outputs: dict[str, Any] = {}
        for name, source in self._output_sources.items():
            x = current_outputs[source]
            if name in self._heads_present:
                head_out = self.output_heads[name](
                    head_state[name], x, context=context
                )
                new_head_state[name] = head_out.next_state
                reg_loss = reg_loss + jp.sum(head_out.regularization_loss)
                outputs[name] = head_out.output
            else:
                new_head_state[name] = head_state[name]
                outputs[name] = x

        new_state = {
            "populations": new_pop_state,
            "connections": new_conn_state,
            "heads": new_head_state,
        }
        return StatefulModuleOutput(new_state, outputs, reg_loss, metrics)

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------

    def initialize_state(self, batch_size: int) -> ModuleState:
        self._assert_finalized()
        pop_state: dict[str, dict] = {}
        for name, pop in self.populations.items():
            entry: dict[str, Any] = {
                "compute": pop.compute.initialize_state(batch_size)
            }
            if pop.max_outgoing_delay > 0:
                entry["buffer"] = jp.zeros(
                    (batch_size, pop.max_outgoing_delay, pop.output_size)
                )
                entry["buffer_idx"] = jp.zeros(batch_size, jp.int32)
            pop_state[name] = entry
        conn_state = [
            conn.transform.initialize_state(batch_size) for conn in self.connections
        ]
        head_state = {
            name: (
                self.output_heads[name].initialize_state(batch_size)
                if name in self._heads_present
                else ()
            )
            for name in self._output_sources
        }
        return {
            "populations": pop_state,
            "connections": conn_state,
            "heads": head_state,
        }

    def reset_state(self, prev_state: ModuleState) -> ModuleState:
        self._assert_finalized()
        prev_pops = prev_state["populations"]
        prev_conns = prev_state["connections"]
        prev_heads = prev_state["heads"]

        new_pops: dict[str, dict] = {}
        for name, pop in self.populations.items():
            entry: dict[str, Any] = {
                "compute": pop.compute.reset_state(prev_pops[name]["compute"])
            }
            if pop.max_outgoing_delay > 0:
                entry["buffer"] = jp.zeros_like(prev_pops[name]["buffer"])
                entry["buffer_idx"] = jp.zeros_like(prev_pops[name]["buffer_idx"])
            new_pops[name] = entry
        new_conns = [
            conn.transform.reset_state(prev_conns[i])
            for i, conn in enumerate(self.connections)
        ]
        new_heads = {
            name: (
                self.output_heads[name].reset_state(prev_heads[name])
                if name in self._heads_present
                else prev_heads[name]
            )
            for name in self._output_sources
        }
        return {
            "populations": new_pops,
            "connections": new_conns,
            "heads": new_heads,
        }

