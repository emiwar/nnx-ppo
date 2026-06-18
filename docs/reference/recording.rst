Recording activations
=====================

Activation recording captures the per-unit outputs of every layer in a
network so you can inspect what the units are doing. It is meant for
**analysis after training**: you record while running a trained policy in
eval mode. The network you trained is never modified.

There are two steps to it: make a recordable copy of your network with
:func:`~nnx_ppo.networks.recording.with_recording`, then read the
activations off each forward call — either one step at a time, or stacked
over a whole episode with
:func:`~nnx_ppo.algorithms.rollout.record_activations_rollout`.

Recording one step
------------------

Wrap the network, run it in eval mode, and pull the activations out of the
returned ``metrics``:

.. code-block:: python

    from nnx_ppo.networks.recording import with_recording, extract_activations

    rec_net = with_recording(network)   # a separate copy; `network` is untouched
    rec_net.eval()                      # deterministic actions for analysis

    state = rec_net.initialize_state(batch_size)
    out = rec_net(state, obs)
    activations = extract_activations(out.metrics)

``activations`` is a nested dict mirroring the network's structure, with one
array per layer. For a standard actor-critic built by
:func:`~nnx_ppo.networks.factories.make_mlp_actor_critic` (a ``Normalizer``
followed by a ``PPOAdapter``) it looks like::

    {
      0: <normalized obs>,              # [B, obs]   the Normalizer
      1: {                              #            the PPOAdapter
        "action": {
          0: <actor layer 0 output>,    # [B, H]
          1: <actor layer 1 output>,    # [B, H]
          2: <distribution params>,     # [B, 2 * action]
          3: {"action": ..., "log_likelihood": ...},   # the sampler
        },
        "value": {
          0: <critic layer 0 output>,   # [B, H]
          1: <critic layer 1 output>,   # [B, H]
          2: <value estimate>,          # [B, 1]
        },
      },
    }

Recording a whole episode
-------------------------

To get a time series, use
:func:`~nnx_ppo.algorithms.rollout.record_activations_rollout`. It makes the
network recordable for you, runs a deterministic rollout, and stacks each
layer's activations over time:

.. code-block:: python

    import jax
    from nnx_ppo.algorithms.rollout import record_activations_rollout

    activations, dones = record_activations_rollout(
        env, network, n_envs=4, max_episode_length=200, key=jax.random.key(0),
    )

Every array now has leading dimensions ``[max_episode_length, n_envs, ...]``.
Environments are **not** reset mid-rollout — each runs a single episode — so
mask out the steps after an env terminates using ``dones`` (the pre-step
"already terminated" flag, shape ``[max_episode_length, n_envs]``):

.. code-block:: python

    import jax.numpy as jp

    keep = 1.0 - dones                          # [T, N]
    actor0 = activations[1]["action"][0]         # [T, N, H]
    masked = actor0 * keep[..., None]            # zero out post-episode steps

.. warning::

   The rollout materialises ``max_episode_length × n_envs × Σ units`` on the
   device. On a small GPU prefer a modest ``n_envs`` (a handful) and/or a
   shorter ``max_episode_length``. For longer or more selective captures, run
   the network in your own Python loop and call ``extract_activations`` on
   each step's ``out.metrics``.

Summary statistics and debugging
--------------------------------

A common use is a quick health check on a new layer — e.g. "do its activations
ever blow up?". Compute the statistic you care about directly from the stacked
arrays, which contain every timestep:

.. code-block:: python

    import jax.numpy as jp

    acts, dones = record_activations_rollout(
        env, network, n_envs=4, max_episode_length=200, key=jax.random.key(0),
    )
    layer = acts[1]["action"][0]                 # [T, N, H]
    print(jp.percentile(jp.abs(layer), jp.array([50.0, 99.0, 100.0])))
    print("peak:", jp.max(jp.abs(layer)))         # catches a one-step spike

Because this sees each step, a transient spike shows up in ``p100`` / the max.

.. note::

   Running a recordable network through ``eval_rollout`` (with
   ``LoggingLevel.NETWORK_METRICS``) *does* emit
   ``eval/net/.../__activation__/p{N}`` keys, but they are **not** what you
   want for catching blow-ups: ``eval_rollout`` averages each metric over the
   episode *before* taking percentiles, so a spike at a single timestep is
   washed out. Those percentiles describe the spread of the *episode-mean*
   activation across units and envs — a cheap "typical magnitude" sanity check,
   not a peak detector. For peaks, stack with ``record_activations_rollout`` as
   above.

How the keys are named
----------------------

Each layer's activation is keyed by its position in the network tree, the
same way state and metrics are keyed:

- ``Sequential`` layers use their **integer index** (``0``, ``1``, ...).
- ``Parallel`` / ``Concat`` / ``PPOAdapter`` use their **string keys**
  (``"action"``, ``"value"``, ...).

A sampler layer records its sampler dict
(``{"action", "log_likelihood"}``), because that is its forward output.

.. note::

   ``Sequential``'s integer keys shift if you insert or reorder layers, so a
   key like ``activations[1]["action"][0]`` is tied to the exact architecture.
   This same fragility affects checkpoint paths and logged metric names; a
   broader review of layer naming is tracked separately (see
   ``docs/_design_notes.md``).

Population graphs
-----------------

A :class:`~nnx_ppo.networks.graph.graph.PopulationGraph` records the
post-activation output of **every population**, not just the ones it exposes
as outputs — its internal populations are usually the interesting part.
``extract_activations`` returns one entry per population, keyed by population
name:

.. code-block:: python

    activations = extract_activations(out.metrics)
    activations["hidden"]   # [B, size] activation of the "hidden" population

This works automatically; ``with_recording`` turns it on. (It is a separate
mechanism because populations are not sub-modules, so they cannot be wrapped
like ordinary layers.)

Under the hood
--------------

``with_recording`` walks the network and wraps every leaf layer in a
:class:`~nnx_ppo.networks.recording.Recorder`. A ``Recorder`` simply calls the
layer it wraps, then attaches that layer's forward ``output`` to the returned
``metrics`` under a reserved key
(:data:`~nnx_ppo.networks.recording.ACTIVATION_KEY`). The output itself is
passed through unchanged, so the wrapped network computes exactly what the
original does — the activation just rides along on the ``metrics`` channel that
containers already propagate to the top-level call.
``extract_activations`` then walks that ``metrics`` tree and pulls out the
values stored under the reserved key.

This is why recording is "off" unless you ask for it — the wrapping lives only
on the copy returned by ``with_recording``; the original network has no
recording code in its path at all. A couple of practical points:

- The copy shares parameters with the original, so it is cheap to make and
  cannot affect training.
- Run it in ``eval()`` mode so action sampling is deterministic and the
  activations reflect the policy's mean behaviour.
