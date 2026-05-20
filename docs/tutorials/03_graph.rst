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

A graph is a set of named **populations** (nodes) connected by
directed **connections** (edges). At each step:

1. For each population in topological order:
2. The population sums all incoming connection outputs (and, if it has
   an ``input_from`` key, the corresponding ``obs[key]`` value). This
   is the population's integrated input.
3. The population's ``compute`` module (default: identity) maps the
   integrated input to a pre-activation output.
4. The population's activation function is applied **once** per step
   to that output, giving the post-activation result. Connections are
   linear by default — activations live on populations, not
   connections, so a node fed by two connections is not double-non-linear.
5. The post-activation output is written into the population's shared
   delay buffer (sized to the longest outgoing delay from that
   population), so future steps can read delayed copies.

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
hidden population, and an output read-out. The hidden population has
a self-loop with ``delay=1`` so it acts like a one-step recurrent
unit.

.. code-block:: python

    from flax import nnx
    from nnx_ppo.networks.feedforward import Dense
    from nnx_ppo.networks.graph import PopulationGraph
    from nnx_ppo.networks.containers import Sequential
    from nnx_ppo.networks.normalizer import Normalizer
    from nnx_ppo.algorithms.adapter import PPOAdapter
    from nnx_ppo.algorithms.distributions import NormalTanhSampler

    rngs = nnx.Rngs(0)

    OBS, H, A = 8, 32, 2

    g = PopulationGraph(rngs)

    # Input layer: reads obs["x"], projects up to H, applies swish.
    g.add_population(
        "input",
        input_size=OBS,
        output_size=H,
        compute=Dense(OBS, H, rngs),
        activation=nnx.swish,
        input_from="x",
    )

    # Recurrent hidden layer: sum-integrates input + own delayed output,
    # passes the sum through swish (activation lives on the population).
    g.add_population("hidden", output_size=H, activation=nnx.swish)
    g.connect("input", "hidden")                 # default Dense(H, H), delay=0
    g.connect("hidden", "hidden", delay=1)       # delayed self-loop

    # Output read-out: separate Dense heads for action params and value.
    g.add_output(
        "action_params", source="hidden",
        head=Dense(H, 2 * A, rngs, kernel_init=nnx.initializers.zeros),
    )
    g.add_output(
        "value", source="hidden",
        head=Dense(H, 1, rngs, kernel_init=nnx.initializers.zeros),
    )

    g.finalize()

A few things to notice:

- ``g.connect("input", "hidden")`` did not need an explicit
  ``transform=``. The default is a linear ``Dense(H, H)`` sized to fit
  ``input``'s ``output_size`` and ``hidden``'s ``input_size``.
- ``g.connect("hidden", "hidden", delay=1)`` is the recurrence.
  Without the ``delay=1`` this would be a delay-0 self-loop and
  :meth:`~nnx_ppo.networks.graph.PopulationGraph.finalize` would
  reject it.
- The hidden population has no explicit ``compute`` — the default is
  identity, which works because its ``input_size`` and ``output_size``
  are both ``H``. The activation lives on the population, so the
  ``input`` and ``hidden`` outputs each go through ``swish`` exactly
  once per step.
- :meth:`~nnx_ppo.networks.graph.PopulationGraph.add_output` declares
  what the graph's forward output dict looks like. Each output is a
  pair of "source population name" and an optional ``head`` module.
  The output dict here is ``{"action_params": ..., "value": ...}``,
  ready to feed a :class:`PPOAdapter`.

Wrap with normalization and the adapter
---------------------------------------

The graph itself is a :class:`StatefulModule`, so you can put it
behind a :class:`~nnx_ppo.networks.normalizer.Normalizer` using
:class:`~nnx_ppo.networks.containers.Sequential`, and then wrap the
whole thing in a :class:`PPOAdapter`:

.. code-block:: python

    inner = Sequential([
        Normalizer({"x": OBS}),
        g,
    ])
    nets = PPOAdapter(
        inner=inner,
        action_specs={"action_params": NormalTanhSampler(rngs, entropy_weight=1e-2)},
        value_specs="value",
    )

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
body part, with afferent connections toward a central "root"
population (``delay=0``) and efferent connections back out
(``delay=1`` to break the cycle). Activation lives on each
population, connections are linear, and the entire topology is
declared up-front. Once :meth:`finalize` is called you can JIT the
whole network and step it like any other :class:`StatefulModule`.

If you need a layer that the existing modules and containers do not
cover — say, your own custom integration rule, or a recurrent cell
the library does not ship — see :doc:`04_custom_module`.

For details on per-connection delays, ``Delay`` as a standalone
layer, and how :class:`Normalizer` interacts with the surrounding
network, see :doc:`../reference/delay_and_normalizer`.
