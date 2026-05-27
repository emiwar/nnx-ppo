``Delay`` and ``Normalizer`` placement
======================================

Both :class:`~nnx_ppo.networks.delay.Delay` and
:class:`~nnx_ppo.networks.normalizer.Normalizer` are ordinary
:class:`~nnx_ppo.networks.types.StatefulModule` s — they can sit
anywhere a layer fits, not just at the top of the network.

``Delay``
---------

:class:`~nnx_ppo.networks.delay.Delay(sample_input, k_steps, initial_value=0.0)`
is a fixed-length circular buffer. The output at time ``t`` is the
input from time ``t - k_steps``. Before the buffer has filled
(``t < k_steps``), the output is ``initial_value``. On
:meth:`reset_state` the buffer is zeroed and the write pointer
restarts.

The shape of the buffer is inferred from the unbatched ``sample_input``
PyTree — pass a single example obs / activation so :class:`Delay`
knows the leaf shapes and dtypes.

Common placements:

**Top-level observation delay.** Wrap the entire network behind a
``Delay`` to give the agent stale observations:

.. code-block:: python

    sample_obs = jax.jit(env.reset)(jax.random.key(0)).obs
    delayed_net = Sequential([Delay(sample_obs, k_steps=5), inner])

**Inside a stack.** ``Delay`` works just like any other layer in
``Sequential``:

.. code-block:: python

    Sequential([encoder, Delay(latent_shape, k_steps=3), decoder])

**As a graph connection transform.** When a graph connection should
carry a *transformed* signal with extra delay, use a small
``Sequential`` for the transform:

.. code-block:: python

    graph.connect(
        "src", "dst",
        transform=Sequential([Dense(H, H, rngs), Delay(jp.zeros(H), k_steps=2)]),
    )

If you only need plain integer-step delay, use the ``delay=`` kwarg
on :meth:`PopulationGraph.connect` instead — it shares a single
buffer per source population. The ``Delay`` module is the right tool
when the delay is per-connection *and* the connection has a custom
transform.

**Recurrent self-loop.** With :class:`PopulationGraph` you do not
need ``Delay`` for recurrence — ``graph.connect(k, k, delay=1)`` is
enough. With containers, build the recurrence by hand using
``Delay`` plus a self-referential ``Sequential``.

``Normalizer``
--------------

:class:`~nnx_ppo.networks.normalizer.Normalizer(shape)` maintains a
running mean and variance (via batched Welford) over the inputs it
sees, and standardises its input using those running stats. The
``shape`` argument is either an ``int`` / ``tuple`` (for a flat
tensor) or a pytree of shapes (for structured observations). Stats
are stored as :class:`nnx.Variable` s; they do *not* receive
gradients.

The forward pass standardises every call. The running stats only
update when :meth:`update_statistics` is called explicitly by the PPO
training loop (after the gradient phase), with the rollout's
``rollout_extras`` history — see :doc:`contexts`.

Top-level placement (typical)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Putting the normalizer at the top of the network is the simplest
case: every observation the network sees first passes through the
normalizer, and the stats it tracks are the raw observations.

.. code-block:: python

    nets = Sequential([
        Normalizer(env.observation_size),
        PPOAdapter(
            action=Sequential([actor, sampler]),
            value=critic,
        ),
    ])

This is what
:func:`~nnx_ppo.networks.factories.make_mlp_actor_critic` does
internally.

Embedded placement
^^^^^^^^^^^^^^^^^^

You can also put a :class:`Normalizer` deeper in the network — behind
a :class:`Delay`, after an encoder, inside a population in a
:class:`PopulationGraph`. The stats it tracks are the activations it
*actually sees* at that point in the network, not the raw
observations.

.. code-block:: python

    Sequential([Delay(sample_obs, k=5), Normalizer(obs_shape), actor])

The normalizer here tracks delayed observations, not fresh ones.

The ``update_statistics`` model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

How can an embedded normalizer see the right inputs? Because on every
forward pass, :class:`Normalizer` emits the input it just standardised
as its ``rollout_extras`` slot. The rollout scan stacks these over T
into ``Transition.rollout_extras``, with structure matching the
network's state tree.

After the gradient phase of each PPO step,
:func:`~nnx_ppo.algorithms.ppo.ppo_step` calls
``network.update_statistics(rollout.rollout_extras)``. The containers
route per child the same way they routed state; each
:class:`Normalizer` receives a ``[T, B, *feat]`` slice of exactly the
inputs it saw during rollout, and folds the batch into its running
mean / M2 / counter via one batched Welford merge. No replay forward
pass is required.

This is why the "no writes to NNX variables that affect the forward
output in ``__call__``, ever" rule from :doc:`contexts` is load-bearing:
it keeps the rollout / loss-replay agreement, and lets statistics flow
through the explicit :meth:`update_statistics` channel rather than as
hidden side effects of a forward pass.

Pytree shapes
^^^^^^^^^^^^^

For dict-structured observations, pass a matching dict of shapes:

.. code-block:: python

    Normalizer({"proprio": 12, "goal": 4})

The forward pass returns a dict with each leaf standardised
independently. Useful when each obs key has a wildly different
scale.

Stats lifecycle
^^^^^^^^^^^^^^^

- Initialisation: stats start at ``mean=0``, ``M2=0``, ``counter=0``.
  Before any update, the normalizer uses a default std of 10 (so it
  effectively passes input through, scaled down to be order-1).
- Updates are accumulated across the entire training run (no
  windowing). To "reset" stats you would construct a new
  :class:`Normalizer` instance.
- Stats are persisted by the checkpointing path along with
  everything else (they are NNX variables).
