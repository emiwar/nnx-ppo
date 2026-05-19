"""Internal population node used by :class:`PopulationGraph`.

A :class:`Population` is a small ``nnx.Module`` that bundles the per-node
specification: declared input/output sizes, the inner compute module that
maps integrated input to pre-activation output, an optional activation,
and (after graph finalisation) the maximum outgoing delay any outgoing
connection requests from this population.

Users do not instantiate :class:`Population` directly — they call
``PopulationGraph.add_population(...)`` which constructs and registers
one. A population in isolation has no semantics: its inputs come from
connections that only exist inside the graph.
"""

from typing import Callable, Optional

from flax import nnx

from nnx_ppo.networks.types import StatefulModule


class Population(nnx.Module):
    """Internal node spec for :class:`PopulationGraph`."""

    def __init__(
        self,
        name: str,
        input_size: int,
        output_size: int,
        compute: StatefulModule,
        activation: Optional[Callable],
        input_from: Optional[str],
    ):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.compute = compute
        self.activation = activation
        self.input_from = input_from
        self.max_outgoing_delay = 0
