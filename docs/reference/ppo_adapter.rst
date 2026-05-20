:class:`PPOAdapter` reference
=============================

:class:`~nnx_ppo.algorithms.adapter.PPOAdapter` is the general-purpose
:class:`~nnx_ppo.networks.types.PPONetwork`. It wraps any
:class:`~nnx_ppo.networks.types.StatefulModule` whose forward output
is a **dict of named heads**, and it knows how to:

- run a named subset of those heads through declared action
  samplers,
- read another named subset as value estimates,
- assemble a :class:`~nnx_ppo.networks.types.PPONetworkOutput`
  bundling the actions, raw actions, log-likelihoods, value
  estimates, regularization loss, metrics, and pre-sampling
  distribution parameters.

Minimal by design
-----------------

The adapter knows about samplers and value heads; everything else
(observation filtering, detached critics, multi-head trunks,
per-head scaling) is a **composition** concern handled by containers
in front of ``inner``. Reach for these patterns instead of asking
the adapter for more:

- **Detached or parallel critic** — combine a graph emitting
  ``{motor_*}`` with a separate MLP critic emitting ``{value_*}``
  through :class:`~nnx_ppo.networks.utils.Merge`.
- **Privileged critic** — let actor and critic branches consume
  different obs slices via two
  :class:`~nnx_ppo.networks.utils.Filter` s inside a :class:`Merge`.
- **Per-head scaling** — wrap a head in
  :class:`~nnx_ppo.networks.utils.Scale` rather than baking the
  factor into a Dense initializer.

See :doc:`../tutorials/02_composition` for worked examples.

Constructor
-----------

.. code-block:: python

    PPOAdapter(
        inner: StatefulModule,
        action_specs: dict[str, ActionSampler],
        value_specs: str | Sequence[str] | dict[str, Any],
    )

``inner``
    A :class:`StatefulModule`. ``inner(state, obs, context=...)``
    must return a :class:`StatefulModuleOutput` whose ``.output`` is
    a dict containing every key in ``action_specs`` and every name
    in ``value_specs``. Other keys in the dict are allowed and
    ignored.

``action_specs``
    Maps inner-output name → :class:`ActionSampler`. The sampler is
    called on ``inner_output[name]``. For a typical Gaussian-tanh
    policy this is a ``(B, 2A)`` ``[mean | log_std]`` tensor.

``value_specs``
    Either a single string (one value head), a sequence of strings
    (multiple named value heads), or a dict whose keys name the
    value heads. The adapter reads each named head and squeezes a
    trailing length-1 axis if present, so a ``(B, 1)`` value head
    becomes a ``(B,)`` value estimate.

The inner-dict contract
-----------------------

The adapter does no per-key processing besides sampler dispatch and
value squeeze. ``inner``'s output dict needs the right keys with the
right shapes:

- For an entry ``"action_params" -> sampler`` in ``action_specs``,
  ``inner_output["action_params"]`` should be the input the sampler
  expects (``(B, 2A)`` for ``NormalTanhSampler``).
- For an entry ``"value"`` in ``value_specs``,
  ``inner_output["value"]`` should be ``(B,)`` or ``(B, 1)``.

How you produce that dict is up to you:
:class:`~nnx_ppo.networks.containers.Parallel` (run separate
sub-modules on the same input → dict),
:class:`~nnx_ppo.networks.containers.Splitter` (slice a flat tensor
into named pieces → dict),
:class:`~nnx_ppo.networks.graph.PopulationGraph` (each
:meth:`add_output` declaration becomes a dict key) — or a custom
:class:`StatefulModule` that emits the dict directly.

Single-head dict-unwrap
-----------------------

If ``action_specs`` has exactly one entry, the resulting
``PPONetworkOutput.actions`` / ``raw_actions`` / ``loglikelihoods``
fields are the bare sampler outputs rather than single-key dicts.
Same for ``value_estimates`` if ``value_specs`` is a string (or a
single-entry sequence/dict).

This means a one-actor / one-critic network produces the same
``PPONetworkOutput`` shape as the legacy ``PPOActorCritic``:

.. code-block:: python

    nets = PPOAdapter(inner, action_specs={"action_params": sampler},
                      value_specs="value")
    state = nets.initialize_state(B)
    state, out = nets(state, obs)
    out.actions.shape          # (B, A)        — bare array
    out.value_estimates.shape  # (B,)          — bare array

With multiple action heads:

.. code-block:: python

    nets = PPOAdapter(
        inner=trunk,
        action_specs={"arm": sampler_arm, "leg": sampler_leg},
        value_specs=["arm_value", "leg_value"],
    )
    state, out = nets(state, obs)
    out.actions                # {"arm": (B, A_arm), "leg": (B, A_leg)}
    out.value_estimates        # {"arm_value": (B,), "leg_value": (B,)}

The PPO loss accepts either shape — it computes GAE per reward key
independently.

Distribution parameters
-----------------------

:class:`~nnx_ppo.networks.types.PPONetworkOutput` carries a
``distribution_params`` field keyed by action-sampler name. Each
entry is the sampler's metrics dict — for
:class:`~nnx_ppo.algorithms.distributions.NormalTanhSampler` this is
``{"mu": ..., "sigma": ...}``.

This is what distillation losses read from. Typical pattern:

.. code-block:: python

    # Teacher in INFERENCE, deterministic samplers (use the mean).
    teacher.eval()
    _, teacher_out = teacher(state, obs, context=Context.INFERENCE)
    teacher_mu = teacher_out.distribution_params["action_params"]["mu"]

    # Student in LOSS_REPLAY for gradients.
    _, student_out = student(state, obs, raw_action=stored_raw,
                              context=Context.LOSS_REPLAY)
    student_mu = student_out.distribution_params["action_params"]["mu"]

    loss = jp.mean((student_mu - teacher_mu) ** 2)

No second forward pass through either network is needed — the
distribution parameters are populated as a side product of the
sampler call.

Carry state
-----------

The adapter's carry state is::

    {"inner": <inner's carry state>,
     "samplers": {name: <each sampler's carry state>}}

Most samplers (including ``NormalTanhSampler``) are stateless, so
the ``samplers`` dict carries empty tuples in practice. The training
loop manages the carry state opaquely.

The ``PPOActorCritic`` shortcut
-------------------------------

:class:`~nnx_ppo.networks.containers.PPOActorCritic` is a thin
convenience subclass of :class:`PPOAdapter` for the standard
one-actor / one-critic case::

    PPOActorCritic(actor, critic, action_sampler, preprocessor=None)

is equivalent to::

    PPOAdapter(
        inner=Sequential([preprocessor, Parallel(action_params=actor, value=critic)])
              if preprocessor is not None
              else Parallel(action_params=actor, value=critic),
        action_specs={"action_params": action_sampler},
        value_specs="value",
    )

The subclass also exposes ``self.actor`` / ``self.critic`` /
``self.action_sampler`` / ``self.preprocessor`` as direct attributes
so parameter logging and inspection code can find them by name.

Use :class:`PPOActorCritic` when you have one actor and one critic.
Drop down to :class:`PPOAdapter` whenever you have multiple action
heads, multiple value heads, or a non-trivial trunk producing the
dict output.

``train()`` / ``eval()``
------------------------

:meth:`nnx.Module.eval` and :meth:`nnx.Module.train` set the
``deterministic`` attribute recursively across the module tree. The
action sampler reads it under ``Context.INFERENCE`` to decide
between sampling and using the mean. Conventional usage:

- Default: ``nets.train()`` — samples stochastically (this is what
  the training loop does between rollouts).
- For deterministic eval / video / deployment: ``nets.eval()``
  before the inference call.

Under ``Context.ROLLOUT``, the sampler always samples
stochastically (training requires exploration regardless of the
``deterministic`` flag). Under ``Context.LOSS_REPLAY`` and
``Context.STATS_UPDATE``, the stored ``raw_action`` is used. So
``train()`` / ``eval()`` only affect ``Context.INFERENCE`` calls.
