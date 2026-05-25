"""Population-graph container.

:class:`PopulationGraph` is a :class:`StatefulModule` that owns a set of
named :class:`Population` nodes and typed :class:`Connection` edges
between them.

Each population has a single ``size``. Connections between populations
default to a linear :class:`Dense` transform sized from source to
destination. Each population sum-integrates its incoming connections
(and optionally the corresponding obs entry, for input populations),
then applies its activation (transfer function) once. Connections
carry an integer ``delay``; ``delay=0`` reads the source's freshly
computed output in the same step (the topological order guarantees
the source has already run), ``delay=k`` reads from ``k`` steps ago
via a per-population shared circular buffer.

The build API is three methods:

* :meth:`add_population(name, size, *, activation=None)` — internal pop.
* :meth:`add_input(name, size, *, input_from=key, activation=None)` —
  reads ``obs[key]`` as the population's integrated input.
* :meth:`add_output(name, size, *, output_to=None, activation=None)` —
  the population's post-activation activation is exposed under
  ``output[output_to or name]`` in the graph's forward output.

Plus :meth:`connect(src, dst, *, transform=None, delay=0, reciprocal=False)`
to wire populations together, and :meth:`finalize()` to validate.

Use :class:`PopulationGraph` when you have a graph with multiple
connections per node, recurrent loops, or per-connection delays. For
straight encoder/decoder stacks ``Sequential`` is simpler.
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


class PopulationGraph(StatefulModule):
    """Declarative population/connection graph as a :class:`StatefulModule`.

    Construction is two-phase:

    1. ``add_population`` / ``add_input`` / ``add_output`` / ``connect``
       to describe the graph.
    2. ``finalize()`` to validate (cycle detection, shape inference,
       buffer sizing) and freeze the graph for forward passes.

    After ``finalize()``, the graph behaves as any other
    :class:`StatefulModule`: it can be a layer in ``Sequential``, an
    inner module of an adapter, etc.
    """

    def __init__(self, rngs: nnx.Rngs):
        self.rngs = rngs
        # Pre-finalize registries (plain Python; promoted to nnx containers
        # in finalize()).
        self._pops: dict[str, Population] = {}
        self._conns: list[Connection] = []
        self._finalized = False
        # Filled in by finalize().
        self._topo_order: tuple[str, ...] = ()
        self._incoming: dict[str, tuple[int, ...]] = {}
        self._output_pops: tuple[tuple[str, str], ...] = ()  # (output_key, pop_name)

    # ------------------------------------------------------------------
    # Build API
    # ------------------------------------------------------------------

    def add_population(
        self,
        name: str,
        size: int,
        *,
        activation: Optional[Callable] = None,
    ) -> None:
        """Register an internal population.

        Args:
          name: Unique identifier.
          size: Size of the population's activation vector. All incoming
            connections must produce vectors of this size after their
            transform.
          activation: Optional transfer function applied elementwise
            *once* after the sum integration of incoming connections.
        """
        self._add_population(
            name=name,
            size=size,
            activation=activation,
            input_from=None,
            output_to=None,
        )

    def add_input(
        self,
        name: str,
        size: int,
        *,
        input_from: str,
        activation: Optional[Callable] = None,
    ) -> None:
        """Register an input population that reads from ``obs[input_from]``.

        The corresponding ``obs[input_from]`` value must have shape
        ``[B, size]`` at call time. The obs value is added to the
        population's integrated input — incoming connections (if any)
        are summed in alongside it before the activation.
        """
        self._add_population(
            name=name,
            size=size,
            activation=activation,
            input_from=input_from,
            output_to=None,
        )

    def add_output(
        self,
        name: str,
        size: int,
        *,
        output_to: Optional[str] = None,
        activation: Optional[Callable] = None,
    ) -> None:
        """Register an output population whose activation is exposed.

        The population's post-activation activation appears in the
        graph's forward output dict under ``output_to`` (or under
        ``name`` if ``output_to`` is None).
        """
        self._add_population(
            name=name,
            size=size,
            activation=activation,
            input_from=None,
            output_to=output_to if output_to is not None else name,
        )

    def _add_population(
        self,
        *,
        name: str,
        size: int,
        activation: Optional[Callable],
        input_from: Optional[str],
        output_to: Optional[str],
    ) -> None:
        self._assert_not_finalized()
        if name in self._pops:
            raise ValueError(f"population {name!r} already exists")
        self._pops[name] = Population(
            name=name,
            size=size,
            activation=activation,
            input_from=input_from,
            output_to=output_to,
        )

    def connect(
        self,
        src: str,
        dst: str,
        *,
        transform: Optional[StatefulModule] = None,
        delay: int = 0,
        reciprocal: bool = False,
    ) -> None:
        """Add a directed connection from ``src`` to ``dst``.

        Args:
          src: Source population name. Must already be registered.
          dst: Destination population name. Must already be registered.
          transform: A :class:`StatefulModule` mapping
            ``[B, src.size] -> [B, dst.size]``. Defaults to a linear
            :class:`Dense` of the appropriate shape.
          delay: Integer step delay. ``0`` reads the source's output
            from the current step; ``k >= 1`` reads from ``k`` steps
            ago. Before the buffer fills, delayed reads return zeros.
          reciprocal: If True, additionally add the reverse connection
            ``dst -> src`` with the same ``delay`` and an independent
            default Dense transform. ``transform`` must be ``None``
            when ``reciprocal`` is True (use two explicit
            :meth:`connect` calls if you need custom transforms in
            both directions).
        """
        if reciprocal and transform is not None:
            raise ValueError(
                "connect(reciprocal=True) requires the default transform; "
                "make two explicit connect() calls if you need custom "
                "transforms in each direction"
            )
        self._add_connection(src, dst, transform=transform, delay=delay)
        if reciprocal:
            self._add_connection(dst, src, transform=None, delay=delay)

    def _add_connection(
        self,
        src: str,
        dst: str,
        *,
        transform: Optional[StatefulModule],
        delay: int,
    ) -> None:
        self._assert_not_finalized()
        if src not in self._pops:
            raise ValueError(f"unknown source population {src!r}")
        if dst not in self._pops:
            raise ValueError(f"unknown destination population {dst!r}")
        if transform is None:
            transform = Dense(
                self._pops[src].size,
                self._pops[dst].size,
                rngs=self.rngs,
            )
        self._conns.append(
            Connection(src=src, dst=dst, transform=transform, delay=delay)
        )

    def finalize(self) -> None:
        """Validate and freeze the graph.

        Performs:
          * delay-0 cycle detection (any cycle is a hard error);
          * topological ordering of populations (delay-0 edges only);
          * per-population ``max_outgoing_delay`` computation;
          * promotion of populations / connections to NNX containers so
            their parameters are tracked.
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

        # Collect output-emitting populations (output_key, pop_name).
        # Order follows add_population insertion order — deterministic.
        self._output_pops = tuple(
            (pop.output_to, name)
            for name, pop in self._pops.items()
            if pop.output_to is not None
        )

        # Promote to nnx-tracked containers.
        self.populations = nnx.Dict(self._pops)
        self.connections = nnx.List(self._conns)

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

        batch_size = jax.tree.leaves(obs)[0].shape[0]
        arange = jp.arange(batch_size)

        new_pop_state: dict[str, dict] = {}
        new_conn_state: list[Any] = [None] * len(self.connections)
        current_outputs: dict[str, jax.Array] = {}
        reg_loss = jp.array(0.0)
        metrics: dict[str, Any] = {}

        # Process populations in topo order so delay-0 reads see freshly
        # computed source outputs.
        for pop_name in self._topo_order:
            pop = self.populations[pop_name]

            integrated = jp.zeros((batch_size, pop.size))
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

            activated = (
                pop.activation(integrated)
                if pop.activation is not None
                else integrated
            )
            current_outputs[pop_name] = activated

            updated: dict[str, Any] = {}
            if pop.max_outgoing_delay > 0:
                buf = pop_state[pop_name]["buffer"]
                idx = pop_state[pop_name]["buffer_idx"]
                new_buf = buf.at[arange, idx].set(activated)
                new_idx = (idx + 1) % pop.max_outgoing_delay
                updated["buffer"] = new_buf
                updated["buffer_idx"] = new_idx
            new_pop_state[pop_name] = updated

        # Connection transforms not visited above keep their old state.
        for i, s in enumerate(conn_state):
            if new_conn_state[i] is None:
                new_conn_state[i] = s

        outputs: dict[str, Any] = {
            output_key: current_outputs[pop_name]
            for output_key, pop_name in self._output_pops
        }

        new_state = {
            "populations": new_pop_state,
            "connections": new_conn_state,
        }
        return StatefulModuleOutput(new_state, outputs, reg_loss, metrics)

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------

    def initialize_state(self, batch_size: int) -> ModuleState:
        self._assert_finalized()
        pop_state: dict[str, dict] = {}
        for name, pop in self.populations.items():
            entry: dict[str, Any] = {}
            if pop.max_outgoing_delay > 0:
                entry["buffer"] = jp.zeros(
                    (batch_size, pop.max_outgoing_delay, pop.size)
                )
                entry["buffer_idx"] = jp.zeros(batch_size, jp.int32)
            pop_state[name] = entry
        conn_state = [
            conn.transform.initialize_state(batch_size)
            for conn in self.connections
        ]
        return {"populations": pop_state, "connections": conn_state}

    def reset_state(self, prev_state: ModuleState) -> ModuleState:
        self._assert_finalized()
        prev_pops = prev_state["populations"]
        prev_conns = prev_state["connections"]

        new_pops: dict[str, dict] = {}
        for name, pop in self.populations.items():
            entry: dict[str, Any] = {}
            if pop.max_outgoing_delay > 0:
                entry["buffer"] = jp.zeros_like(prev_pops[name]["buffer"])
                entry["buffer_idx"] = jp.zeros_like(
                    prev_pops[name]["buffer_idx"]
                )
            new_pops[name] = entry
        new_conns = [
            conn.transform.reset_state(prev_conns[i])
            for i, conn in enumerate(self.connections)
        ]
        return {"populations": new_pops, "connections": new_conns}
