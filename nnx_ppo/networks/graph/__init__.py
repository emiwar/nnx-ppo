"""Population-graph network container.

Public surface: :class:`PopulationGraph`. :class:`Population` and
:class:`Connection` are internal node/edge specs constructed by the
graph's build methods (``add_population`` / ``connect``).
"""

from nnx_ppo.networks.graph.connection import Connection
from nnx_ppo.networks.graph.graph import PopulationGraph
from nnx_ppo.networks.graph.population import Population

__all__ = ["PopulationGraph", "Population", "Connection"]
