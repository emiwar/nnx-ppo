"""Internal population node used by :class:`PopulationGraph`.

A :class:`Population` is a small ``nnx.Module`` that bundles the per-node
specification: declared size, optional activation (transfer function),
and obs / output routing annotations (``input_from`` for input pops,
``output_to`` for output pops). After graph finalisation it also
carries the maximum outgoing delay any outgoing connection requests
from this population, which sizes its shared output buffer.

Users do not instantiate :class:`Population` directly — they call
``PopulationGraph.add_population(...)`` / ``add_input(...)`` /
``add_output(...)`` which construct and register one. A population in
isolation has no semantics: its inputs come from connections that
only exist inside the graph.
"""

from typing import Callable, Optional

from flax import nnx


class Population(nnx.Module):
    """Internal node spec for :class:`PopulationGraph`."""

    def __init__(
        self,
        name: str,
        size: int,
        activation: Optional[Callable],
        input_from: Optional[str],
        output_to: Optional[str],
    ):
        self.name = name
        self.size = size
        self.activation = activation
        self.input_from = input_from
        self.output_to = output_to
        self.max_outgoing_delay = 0
