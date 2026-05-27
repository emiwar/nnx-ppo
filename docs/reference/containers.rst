Containers
==========

The :mod:`nnx_ppo.networks.containers` module provides the four core
composition primitives: :class:`Sequential`, :class:`Parallel`,
:class:`Concat`, and :class:`Splitter`. Every container is itself a
:class:`~nnx_ppo.networks.types.StatefulModule`, so they nest freely.

Each container threads three things through its children:

- **carry state** — sliced per-child the same way state is structured
  (list for Sequential, dict for Parallel/Concat, leaf for Splitter);
- **rollout_extras** — same routing as state;
- **regularization loss & metrics** — accumulated from all children
  and surfaced on the container's own ``StatefulModuleOutput``.

Sequential
----------

Chain layers. ``Sequential([a, b, c])`` runs ``a → b → c`` and routes
each layer's carry state through a list keyed by position.

.. code-block:: python

    from nnx_ppo.networks.containers import Sequential

    stack = Sequential([encoder, body, head])
    state = stack.initialize_state(batch_size)
    out = stack(state, obs)
    # out.next_state == [encoder.next_state, body.next_state, head.next_state]

Sequential is the workhorse: most "chains of layers" are Sequentials.
Use it as the action / value port of a :class:`PPOAdapter`, as the
inner body of a Parallel branch, etc.

Parallel
--------

Run several named sub-modules on the **same** input. Returns a dict
keyed by sub-module name.

.. code-block:: python

    from nnx_ppo.networks.containers import Parallel

    # Same input fed to every component.
    branches = Parallel(action_params=actor, value=critic)
    out = branches((), obs)
    # out.output == {"action_params": <actor_out>, "value": <critic_out>}

Typical use: assemble per-key heads on top of a shared trunk, e.g.
per-body-module value heads (``Parallel({k: Dense(H, 1) for k in keys})``)
fed by the same trunk features.

Concat
------

Per-key dispatch + concat: dict input, single-tensor output. Each
named sub-module sees the upstream's same-named entry and the
per-component outputs are concatenated along the last axis.

.. code-block:: python

    from nnx_ppo.networks.containers import Concat

    # Per-stream encoders → single concatenated feature vector.
    encoders = Concat(proprio=proprio_encoder, goal=goal_encoder)
    out = encoders(state, {"proprio": (B, P), "goal": (B, G)})
    # out.output : (B, H_proprio + H_goal)

Use at the *start* of a stack when the observation is structured.

Splitter
--------

Inverse of :class:`Concat`. Takes a flat tensor and splits it into
named slices along the last axis.

.. code-block:: python

    from nnx_ppo.networks.containers import Splitter

    s = Splitter(action_params=2 * action_size, value=1)
    out = s((), x)  # x : (B, 2A + 1)
    # out.output == {"action_params": (B, 2A), "value": (B, 1)}

Use at the *end* of a stack to turn a flat head into a dict output an
adapter / downstream can route to samplers and value specs. With a
single keyword (``Splitter(action_params=N)``) it simply relabels the
input as a dict, taking the first N features.

The slices are taken in keyword-argument insertion order. The sum of
declared sizes must not exceed the input's last-axis size; any excess
input features are silently ignored, matching plain slicing semantics.

Construction forms
------------------

:class:`Concat` and :class:`Parallel` accept *either* keyword arguments
*or* a positional dict, e.g.::

    Parallel({pop: head for pop in POPULATIONS})

Use the positional form when keys aren't valid Python identifiers or
when you're building the container from a comprehension.

Writing your own container
--------------------------

The four shipped containers are not load-bearing on the rest of the
library — they are ordinary :class:`StatefulModule` s that thread
plumbing through their children. If none of them captures your
topology you can write your own. The contract has four parts.

1. Be a valid Flax NNX module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A container holds child modules and their parameters. To make NNX
trace those parameters, every sub-module must be either a direct
attribute or sit inside an NNX-tracked container. For a **fixed** set
of children, plain attribute assignment is enough::

    class TwoChild(StatefulModule):
        def __init__(self, a, b):
            self.a = a
            self.b = b

For a **variable-size** list or dict of children, wrap them in
`flax.nnx.List
<https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/helpers.html#flax.nnx.List>`_
or `flax.nnx.Dict
<https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/helpers.html#flax.nnx.Dict>`_.
A plain Python ``list`` / ``dict`` of modules will **not** be traced
by NNX and your parameters will silently disappear from the trainable
set::

    from flax import nnx

    class MyMap(StatefulModule):
        def __init__(self, modules: dict[str, StatefulModule]):
            self.components = nnx.Dict(modules)   # NNX-tracked

This is what every shipped container does internally (``Sequential``
uses :class:`nnx.List`; ``Concat`` / ``Parallel`` use :class:`nnx.Dict`).

2. Forward state and ``rollout_extras`` per child
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``__call__``, slice the incoming ``state`` and ``rollout_extras``
per child the same way, call each child, and reassemble both into
output containers shaped exactly like the input. A typical loop::

    def __call__(self, state, x, rollout_extras=None):
        new_state, new_extras = {}, {}
        ...
        for name, child in self.components.items():
            child_extras = None if rollout_extras is None else rollout_extras[name]
            out = child(state[name], <child input>, child_extras)
            new_state[name] = out.next_state
            new_extras[name] = out.rollout_extras
            ...

The crucial invariant is that the *shape* of returned ``next_state``
mirrors the shape of the incoming ``state``, and the same for
``rollout_extras``. If a caller passes ``rollout_extras=None``, you
pass ``None`` to every child (each child knows how to fall back). If a
caller passes a non-``None`` value, the structure must match what your
container emitted on a previous step.

You also need to implement :meth:`initialize_state` and (if your
children carry resettable state) :meth:`reset_state` with the same
shape — for the same reasons.

3. Aggregate ``regularization_loss`` and ``metrics``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each child's :attr:`StatefulModuleOutput.regularization_loss` should
flow up to your container's output. In most cases this is just a sum::

    regularization_loss = jp.array(0.0)
    for ...:
        out = child(...)
        regularization_loss += out.regularization_loss

If your container has a reason to scale or otherwise transform a
child's regularization (e.g. a hypothetical container that runs
children alternately and only counts the active one's loss), do that
here — but the default is a plain sum.

``metrics`` are typically collected into a per-child dict
``{child_name: out.metrics}`` so downstream logging can disambiguate.
:class:`Sequential` uses positional integer keys; the named-dispatch
containers use string keys matching their children.

4. Route ``update_statistics`` per child
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stats accumulation runs *outside* ``__call__``. Override
:meth:`update_statistics` to route the rollout's stacked
``rollout_extras`` per child analogously to how ``__call__`` routes
the per-step extras::

    def update_statistics(self, rollout_extras):
        for name, child in self.components.items():
            child.update_statistics(rollout_extras[name])

Without this override the default implementation is a no-op for the
container, which means any stats-bearing module nested inside it will
never see its data and silently fail to update. This is the most
common mistake when writing a new container — easy to forget because
every other rule fails loudly while this one fails silently.

Worked example
~~~~~~~~~~~~~~

The shipped :class:`Parallel` is the smallest container that exercises
all four rules; reading [its
source](../_modules/nnx_ppo/networks/containers.html) is the fastest
way to see the pattern end-to-end (~50 LoC). :class:`Sequential` is
similar but lists rather than dicts; the per-key dispatch containers
:class:`~nnx_ppo.networks.utils.Map` and
:class:`~nnx_ppo.networks.utils.Merge` are also good references.
