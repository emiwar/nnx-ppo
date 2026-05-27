The ``rollout_extras`` channel
==============================

PPO runs the network twice per training iteration: once during *rollout*
to drive the environment and collect data, then once per minibatch
during *loss replay* to compute gradients on the recorded observations.
A handful of modules (action samplers, normalizers) need to communicate
between these two passes — the sampler needs to recompute the
log-likelihood of the actually-taken action under updated weights; the
normalizer needs the rollout's activations to fold into its running
statistics.

``rollout_extras`` is the channel that carries this communication. It
is a pytree shaped exactly like the network's ``state`` tree. Each
container routes children's extras the same way it routes their state.

The three phases
----------------

Rollout
    Driving the environment to collect data.
    :func:`~nnx_ppo.algorithms.rollout.unroll_env` is the entry point.
    Callers pass ``rollout_extras=None``. Each module *emits* its
    replay snapshot into the returned
    ``StatefulModuleOutput.rollout_extras``. The rollout scan stacks
    these over T into ``Transition.rollout_extras``.

Loss replay
    Re-running the rollout to compute the PPO loss and its gradient.
    Threaded into the scan body of
    :func:`~nnx_ppo.algorithms.ppo.ppo_loss`. The per-step
    ``rollout_extras`` slice from the stored ``Transition`` is fed back
    in. Modules that need it (notably action samplers) consume it to
    reproduce the actually-taken action's log-likelihood under the
    current (updated) policy.

Inference
    Anything outside of data collection or training:
    :func:`~nnx_ppo.algorithms.rollout.eval_rollout`, debugging,
    ad-hoc forward passes. Callers pass nothing. Modules still emit
    extras but the caller drops them on the floor.

A module that needs to distinguish "fresh sample" from "use stored
value" reads ``if rollout_extras is None``. There is no ``Context``
enum.

How containers route ``rollout_extras``
---------------------------------------

Every container in :mod:`nnx_ppo.networks.containers`
(``Sequential``, ``Parallel``, ``Concat``, ``Splitter``) and
:mod:`nnx_ppo.networks.utils` (``Flattener``, ``Filter``, ``Scale``,
``Merge``, ``Map``) plus :class:`~nnx_ppo.networks.delay.Delay`,
:class:`~nnx_ppo.networks.graph.PopulationGraph`, and
:class:`~nnx_ppo.networks.adapter.PPOAdapter` accepts ``rollout_extras``
as the third positional argument. Each container slices its incoming
``rollout_extras`` per child the same way it slices ``state``, calls
each child, and reassembles the emitted extras into a tree mirroring
``state``.

Leaf modules either:

- ignore ``rollout_extras`` and emit ``None`` (most layers — Dense,
  LSTM, Filter, Flattener, Scale, Splitter, Delay, VariationalBottleneck),
- or use it (Normalizer, ActionSampler).

Sampler replay rule
-------------------

A module that produces a *sample whose log-likelihood will be
evaluated under updated weights* — i.e. an action sampler in PPO —
**must** store the sample in ``rollout_extras``. RNG-in-state is wrong
because the same RNG under updated weights would produce a *different*
sample, but PPO needs the log-likelihood of the *actually-taken*
action under the new policy.

RNG-in-state is only valid for modules whose sample is consumed as a
forward activation (reparameterised gradient), like
:class:`~nnx_ppo.networks.variational.VariationalBottleneck`. See
:doc:`randomness` for the per-env-RNG-in-carry-state pattern.

The write rule
--------------

A forward pass through any :class:`StatefulModule` must be fully
reproducible — gradients computed during loss replay are gradients for
the activations that drove the environment during rollout. Concretely:

    **No writes to NNX variables that affect the forward output in
    ``__call__``, ever.**

Stats-bearing modules accumulate state by overriding
:meth:`update_statistics`, called once per training step *after* the
gradient update with the full rollout's stacked ``rollout_extras``
history.

Per-module behaviour table
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 38 38

   * - Module
     - rollout_extras passed in
     - rollout_extras emitted
   * - ``Normalizer``
     - ignored — normalise with live ``mean`` / ``M2`` / ``counter``
     - the input ``x`` (every call) — consumed by
       :meth:`update_statistics`
   * - ``ActionSampler`` (e.g. ``NormalTanhSampler``)
     - if not ``None``: use as ``raw_action`` and compute the
       log-likelihood under current policy
     - the freshly-sampled ``raw_action`` (every call)
   * - ``VariationalBottleneck`` / ``AR1VariationalBottleneck``
     - ignored — RNG is in carry state; reparameterised gradient
     - ``None``
   * - everything else (``Dense``, ``LSTM``, containers, ``Delay``,
       ``PopulationGraph``)
     - threaded to children
     - ``None`` at leaves; assembled tree at containers
