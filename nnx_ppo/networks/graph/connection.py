"""Internal edge between two populations in a :class:`PopulationGraph`.

A :class:`Connection` carries a single source population's post-activation
output to a single destination population's integrated input. The
``transform`` (a :class:`StatefulModule`, typically a plain :class:`Dense`)
maps from the source's ``output_size`` to the destination's
``input_size``; ``delay`` shifts the signal by an integer number of
steps (``0`` = same-step).

Users do not instantiate :class:`Connection` directly — they call
``PopulationGraph.connect(...)``.
"""

from flax import nnx

from nnx_ppo.networks.types import StatefulModule


class Connection(nnx.Module):
    """Internal edge spec for :class:`PopulationGraph`."""

    def __init__(
        self,
        src: str,
        dst: str,
        transform: StatefulModule,
        delay: int,
    ):
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")
        self.src = src
        self.dst = dst
        self.transform = transform
        self.delay = delay
