Custom networks 2: graph networks
=================================

The containers in :doc:`02_composition` are enough for any **stack** of
layers — even with branching, the topology stays a tree. They start to
hurt when the topology has:

- multiple connections feeding the same node (the node needs to
  *integrate* incoming signals, not just receive one);
- recurrent loops between nodes;
- per-connection delays.

These are common in modular controllers (one population per body
part, communicating across joints) and in GNN-style message-passing
networks. :class:`~nnx_ppo.networks.graph.PopulationGraph` is a
:class:`StatefulModule` that gives you a declarative API for that
class of network.

What :class:`PopulationGraph` does
----------------------------------

A graph is a set of named **populations** (nodes), each with a single
``size``, connected by directed **connections** (edges) with default
linear :class:`Dense` transforms. At each step:

1. For each population in topological order:
2. The population sums all incoming connection outputs. If it was
   registered with :meth:`add_input`, the corresponding ``obs[key]``
   value is added too. This is the population's integrated input.
3. The population's activation function (if any) is applied **once**
   to the integrated input, giving the post-activation result.
   Connections are linear; activations live on populations — so a node
   fed by two connections is not double-non-linear.
4. The post-activation output is written into the population's shared
   delay buffer (sized to the longest outgoing delay from that
   population), so future steps can read delayed copies.

Populations registered with :meth:`add_output` have their
post-activation value emitted in the graph's forward output dict
(under ``output_to`` if set, else under the population's own name).

Delays are first-class. Each :meth:`~nnx_ppo.networks.graph.PopulationGraph.connect`
takes a ``delay`` argument. ``delay=0`` reads the source's
freshly-computed output in the *same* step (the topological order
guarantees that the source has already run); ``delay=k`` reads the
output from ``k`` steps ago. Before the buffer has filled, delayed
reads return zeros.

A ``delay=0`` cycle is a hard error: it would force a node to read
its own output from the current step, which is undefined. Break the
cycle with a ``delay=1`` self-loop or a ``delay=1`` back-edge.

A worked example
----------------

The smallest non-trivial graph: an input population, a recurrent
hidden population, and two output read-outs. The hidden population
has a self-loop with ``delay=1`` so it acts like a one-step recurrent
unit.

.. code-block:: python

    from flax import nnx
    from nnx_ppo.networks.graph import PopulationGraph
    from nnx_ppo.networks.containers import Sequential
    from nnx_ppo.networks.normalizer import Normalizer
    from nnx_ppo.networks.adapter import PPOAdapter
    from nnx_ppo.networks.sampling_layers import NormalTanhSampler

    rngs = nnx.Rngs(0)

    OBS, H, A = 8, 32, 2

    g = PopulationGraph(rngs)

    # Input population: reads obs["x"]. Same size as obs["x"].
    g.add_input("in_x", size=OBS, input_from="x")

    # Recurrent hidden population: sum-integrates input + own delayed
    # output, passes the sum through swish.
    g.add_population("hidden", size=H, activation=nnx.swish)
    g.connect("in_x", "hidden")                  # default Dense(OBS, H)
    g.connect("hidden", "hidden", delay=1)       # delayed self-loop

    # Output populations: action-params head and value head, each
    # exposed in the graph's forward output under its own name.
    g.add_output("action_params", size=2 * A)
    g.connect("hidden", "action_params")         # default Dense(H, 2A)

    g.add_output("value", size=1)
    g.connect("hidden", "value")                 # default Dense(H, 1)

    g.finalize()

A few things to notice:

- :meth:`add_input` registers a population that reads ``obs["x"]``
  directly. Its size must match the obs entry it consumes; downstream
  connections handle any size change (in this case
  ``Dense(OBS, H)``).
- ``g.connect("in_x", "hidden")`` does not need an explicit
  ``transform=``. The default is a linear ``Dense(in_x.size,
  hidden.size)``.
- ``g.connect("hidden", "hidden", delay=1)`` is the recurrence.
  Without the ``delay=1`` this would be a delay-0 self-loop and
  :meth:`~nnx_ppo.networks.graph.PopulationGraph.finalize` would
  reject it.
- :meth:`add_output` registers a population whose post-activation
  activation appears in the graph's forward output dict. With no
  explicit ``output_to``, the key in the output dict is the
  population's own name.
- Both output populations have no activation — they're linear
  read-outs.

The graph's forward output is
``{"action_params": (B, 2A), "value": (B, 1)}``, ready to feed a
:class:`PPOAdapter`.

Wrap with normalization and the adapter
---------------------------------------

The graph itself is a :class:`StatefulModule`, so you can put it
behind a :class:`~nnx_ppo.networks.normalizer.Normalizer` using
:class:`~nnx_ppo.networks.containers.Sequential`, and then wrap the
whole thing in a :class:`PPOAdapter`:

.. code-block:: python

    from nnx_ppo.networks.utils import Filter, Map

    sampler = NormalTanhSampler(rngs, entropy_weight=1e-2)
    nets = Sequential([
        Normalizer({"x": OBS}),
        g,                                  # emits {"action_params": ..., "value": ...}
        PPOAdapter(
            # Each port sees the graph's full dict and picks what it needs.
            # `Map` dispatches per key and drops keys it doesn't consume.
            action=Map({"action_params": sampler}),
            value=Filter({"value": "value"}),
        ),
    ])

The resulting ``PPONetworkOutput.actions`` is ``{"action_params": (B, A)}``
— a single-key dict because the graph names every output. For
the per-body-module case in the next section, you would have one
sampler per population name in the ``action`` port's ``Map``.

You can now feed this network to :func:`~nnx_ppo.algorithms.ppo.train_ppo`
exactly as in :doc:`01_quickstart`, provided your env's observation
is a dict containing key ``"x"``.

Per-population delay buffers
----------------------------

A node's outgoing connections may have different delays. The graph
allocates a single circular buffer per population, sized to the
longest outgoing delay from that population. All delayed connections
sharing that source read from the same buffer at their own offset —
so the cost of a delayed connection is the connection's own
parameters, plus a constant-size buffer per source population. (No
duplicate storage per connection.)

What you'd build next
---------------------

The toy graph above has one input population and one hidden
population. Real modular networks typically have one population per
body part, with reciprocal connections to a central "root"
population::

    g.connect("hand_L", "arm_L", delay=1, reciprocal=True)
    g.connect("arm_L",  "root",  delay=1, reciprocal=True)
    g.connect("torso",  "root",  delay=1, reciprocal=True)
    ...

``reciprocal=True`` is sugar for two :meth:`connect` calls (one in
each direction, each with its own default Dense and the same delay).
Activation lives on each population, connections are linear, and the
entire topology is declared up-front. Once :meth:`finalize` is called
you can JIT the whole network and step it like any other
:class:`StatefulModule`.

If you need a layer that the existing modules and containers do not
cover — say, your own custom integration rule, or a recurrent cell
the library does not ship — see :doc:`04_custom_module`.

For details on per-connection delays, ``Delay`` as a standalone
layer, and how :class:`Normalizer` interacts with the surrounding
network, see :doc:`../reference/delay_and_normalizer`.
