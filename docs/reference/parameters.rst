Training parameters
===================

Every knob exposed by :func:`~nnx_ppo.algorithms.ppo.train_ppo` lives in
one of four dataclasses: :class:`~nnx_ppo.algorithms.config.PPOConfig`
(the core algorithm),
:class:`~nnx_ppo.algorithms.config.EvalConfig` (evaluation rollouts),
:class:`~nnx_ppo.algorithms.config.VideoConfig` (video rendering), and
:class:`~nnx_ppo.algorithms.config.TrainConfig` (the wrapper that
bundles them with a seed). This page lists every field with its
default and meaning.

``PPOConfig``
-------------

.. py:currentmodule:: nnx_ppo.algorithms.config

Core PPO algorithm parameters. Field defaults shown in parentheses.

``n_envs`` (``256``)
    Number of environments rolled out in parallel per training
    iteration. The total batch size per iteration is
    ``n_envs * rollout_length``. Larger values give lower-variance
    gradient estimates at the cost of GPU memory.
``rollout_length`` (``20``)
    Number of environment steps collected per env, per iteration.
``total_steps`` (``512_000``)
    Total cumulative environment steps to train for. Training stops
    after the iteration that pushes the cumulative count past this
    threshold.
``gae_lambda`` (``0.95``)
    Generalised-advantage-estimation trace decay parameter. Higher
    values (closer to 1) weight long-horizon returns more.
``discounting_factor`` (``0.99``)
    Reward discount γ.
``clip_range`` (``0.2``)
    PPO probability-ratio clip range ε. The actor objective clips the
    importance ratio to ``[1 - ε, 1 + ε]``.
``learning_rate`` (``1e-4``)
    Adam step size shared by the actor and critic.
``normalize_advantages`` (``True``)
    Whether to standardise advantages to zero mean / unit variance
    within each minibatch before computing the actor loss.
``combine_advantages`` (``False``)
    For multi-reward / multi-objective networks: if ``True``, sum the
    per-reward advantages into a single scalar advantage before the
    actor update. If ``False``, advantages stay keyed by reward name
    and the actor sees a multi-objective gradient.
``n_epochs`` (``4``)
    Passes over each rollout batch during the gradient phase.
``n_minibatches`` (``4``)
    Number of minibatches each rollout batch is split into per epoch.
    Total gradient steps per iteration is
    ``n_epochs * n_minibatches``.
``critic_loss_weight`` (``1.0``)
    Scalar coefficient on the critic (value) loss term in the
    combined PPO loss.
``gradient_clipping`` (``None``)
    If set, global-norm gradient clipping at this value. ``None``
    means no clipping.
``weight_decay`` (``None``)
    If set, AdamW-style weight decay coefficient. ``None`` uses plain
    Adam.
``logging_level`` (``LoggingLevel.LOSSES``)
    Which families of metrics to compute and forward to ``log_fn``.
    See :doc:`logging` for the full set of flags.
``logging_percentiles`` (``None``)
    Percentile reduction for vector-valued metrics (e.g. per-env
    rewards). ``None`` means only mean / std are reported; passing
    ``(0, 25, 50, 75, 100)`` adds min / quartiles / max. See
    :doc:`logging`.

``EvalConfig``
--------------

Configures the periodic evaluation rollouts performed during
training.

``enabled`` (``True``)
    Whether to run any evaluation. ``False`` disables eval entirely;
    the other fields are then ignored.
``every_steps`` (``50_000``)
    Approximate interval between eval runs, measured in cumulative
    env steps. An eval is triggered on the first iteration whose
    cumulative step count crosses a multiple of this value.
``n_envs`` (``64``)
    Number of envs stepped in parallel during an eval rollout.
``max_episode_length`` (``1000``)
    Episode cutoff for eval. Useful for envs that do not terminate on
    their own.
``logging_level`` (``LoggingLevel.BASIC``)
    Which metric families to include in eval results. ``BASIC`` (the
    default) is equivalent to ``LOSSES``.
``logging_percentiles`` (``(0, 25, 50, 75, 100)``)
    Percentile reduction for eval metrics (per-env episode reward,
    episode length).

``VideoConfig``
---------------

Configures optional video recording of eval rollouts.

``enabled`` (``False``)
    Whether to record videos. Requires ``video_fn`` to be passed to
    :func:`train_ppo` and an env that supports
    ``env.render(trajectory, **render_kwargs)``.
``every_steps`` (``200_000``)
    Approximate interval between video captures.
``episode_length`` (``1000``)
    Length of the rendered episode in env steps.
``render_kwargs`` (``{"height": 480, "width": 640}``)
    Forwarded verbatim to the env's ``render(...)`` call. Add camera
    selection, label overlays, and other env-specific options here.

``TrainConfig``
---------------

Wraps the three sub-configs plus a global seed and checkpoint
cadence.

``ppo`` (``PPOConfig()``)
    The :class:`PPOConfig` instance.
``eval`` (``EvalConfig()``)
    The :class:`EvalConfig` instance.
``video`` (``VideoConfig()``)
    The :class:`VideoConfig` instance.
``seed`` (``17``)
    Master seed for the JAX RNG streams used by the training loop —
    rollout RNG, minibatch shuffling, eval reset keys. Match this to
    the ``nnx.Rngs(seed)`` you used to build the network if you want
    fully reproducible runs.
``checkpoint_every_steps`` (``500_000``)
    Interval between checkpoint writes when a ``checkpoint_fn`` is
    passed to :func:`train_ppo`. See :doc:`checkpointing`.
