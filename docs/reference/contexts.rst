The ``Context`` enum
====================

:class:`~nnx_ppo.networks.types.Context` is the lifecycle marker
threaded as a keyword-only argument through every
:class:`~nnx_ppo.networks.types.StatefulModule` ``__call__``. Each
forward pass declares "what phase of training am I in", and the few
modules whose behaviour depends on the phase (samplers, normalizers,
variational layers) read this argument to decide.

The four contexts
-----------------

``Context.ROLLOUT``
    Driving the environment to collect data.
    :func:`~nnx_ppo.algorithms.rollout.unroll_env` uses this by
    default.

``Context.LOSS_REPLAY``
    Re-running the rollout to compute the PPO loss and its gradient.
    Threaded into the scan body of
    :func:`~nnx_ppo.algorithms.ppo.ppo_loss`. Action samplers receive
    the stored ``raw_action`` from the rollout so the replay
    reproduces exactly the activations produced during data
    collection.

``Context.INFERENCE``
    Anything outside of data collection or training:
    :func:`~nnx_ppo.algorithms.rollout.eval_rollout`,
    distillation teacher calls, debugging, ad-hoc forward passes.
    This is also the default value of ``context`` for
    :meth:`StatefulModule.__call__`, so forgetting to pass
    ``context=...`` never accidentally collects training data or
    updates statistics.

``Context.STATS_UPDATE``
    A post-loss replay of the rollout whose only purpose is to fold
    rollout activations into running statistics
    (:class:`~nnx_ppo.networks.normalizer.Normalizer` is the canonical
    consumer). Set by
    :func:`~nnx_ppo.algorithms.ppo._replay_rollout_for_stats` and
    discarded as soon as the replay finishes.

Threading rule
--------------

Every container in :mod:`nnx_ppo.networks.containers`
(``Sequential``, ``Parallel``, ``Concat``, ``Splitter``) and
:mod:`nnx_ppo.networks.utils` (``Flattener``, ``Filter``, ``Scale``,
``Merge``) plus :class:`~nnx_ppo.networks.delay.Delay`,
:class:`~nnx_ppo.networks.graph.PopulationGraph`, and
:class:`~nnx_ppo.algorithms.adapter.PPOAdapter`
takes ``context`` as a keyword-only argument and passes it through to
every child it calls. If you write your own container, do the same.

Leaf modules either:

- ignore ``context`` (most layers — Dense, LSTM, ...), or
- read it to switch behaviour (Normalizer, ActionSampler,
  VariationalBottleneck).

The write rule
--------------

The library guarantees that a forward pass is reproducible across
rollout and loss replay — gradients computed in ``LOSS_REPLAY`` are
gradients for the activations that drove the environment in
``ROLLOUT``. Concretely:

    **No writes to NNX variables that affect the forward output,
    unless** ``context == Context.STATS_UPDATE``.

What counts as a write that affects forward output:

- Mutating an :class:`nnx.Param` (would change a future
  ``LOSS_REPLAY`` relative to the matching ``ROLLOUT``).
- Mutating an :class:`nnx.Variable` whose value is read in
  ``__call__``'s return path.

What does *not* count:

- Pure read access to NNX variables.
- Advancing an :class:`nnx.Rngs` stream during ``__init__`` for
  parameter initialisation. That happens once, before any forward
  pass, and is unaffected by context.

A common pitfall — advancing an RNG inside :meth:`__call__` by
reading a class-level :class:`nnx.Variable` key — *does* count as a
forbidden write, and silently produces wrong gradients. See
:doc:`randomness` for the correct pattern (one RNG per env, carried
in the module's carry state).

Per-module behaviour table
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 22 22 22 22

   * - Module
     - ROLLOUT
     - LOSS_REPLAY
     - INFERENCE
     - STATS_UPDATE
   * - ``Normalizer``
     - normalize using live ``mean`` / ``M2`` / ``counter``;
       no writes
     - same as ROLLOUT
     - same as ROLLOUT
     - normalize, **and** update ``mean`` / ``M2`` / ``counter``
       with a per-step Welford step
   * - ``ActionSampler`` (e.g. ``NormalTanhSampler``)
     - sample stochastically (or use mean if ``deterministic``)
     - use stored ``raw_action``; RNG is still advanced so any
       downstream stochastic layers stay in lockstep
     - per-instance ``deterministic`` flag decides
     - use stored ``raw_action`` (same as LOSS_REPLAY)
   * - ``VariationalBottleneck`` / ``AR1VariationalBottleneck``
     - sample using the carried per-env RNG (advances carry)
     - sample using the carried per-env RNG (reproduces rollout
       because the carry starts at the same value and operations are
       deterministic)
     - sample using the carried per-env RNG
     - sample using the carried per-env RNG
   * - everything else (``Dense``, ``LSTM``, containers, ``Delay``,
       ``PopulationGraph``)
     - no context-dependent behaviour
     -
     -
     -

Where each context is set
-------------------------

- :func:`~nnx_ppo.algorithms.rollout.unroll_env` —
  ``Context.ROLLOUT``.
- :func:`~nnx_ppo.algorithms.rollout.eval_rollout` and
  :func:`~nnx_ppo.algorithms.rollout.eval_rollout_for_render_scan` —
  ``Context.INFERENCE``.
- :func:`~nnx_ppo.algorithms.ppo.ppo_loss` —
  ``Context.LOSS_REPLAY``.
- :func:`~nnx_ppo.algorithms.ppo._replay_rollout_for_stats`, called
  by :func:`~nnx_ppo.algorithms.ppo.ppo_step` after the gradient
  phase — ``Context.STATS_UPDATE``.

A network that you call yourself (a distillation teacher, an
analysis script, the inference path of a deployed agent) should pass
``Context.INFERENCE`` — that is what the default lands on if you
forget.

Stochastic layers
-----------------

The rule-of-writes above keeps deterministic operations consistent
across ROLLOUT and LOSS_REPLAY, but stochastic operations need an
RNG that is also consistent across the two passes. The pattern that
guarantees this — one RNG per env, carried in the module's carry
state rather than held as a class-level variable — is documented in
:doc:`randomness`. The ``ActionSampler`` and ``VariationalBottleneck``
rows of the table above are the canonical examples.
