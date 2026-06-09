Logging
=======

:func:`~nnx_ppo.algorithms.ppo.train_ppo` exposes training progress
through a ``log_fn`` callback. This page documents the callback
signature, the shape of the metrics dict, the
:class:`~nnx_ppo.algorithms.types.LoggingLevel` flag enum that
selects which metric families are computed, and the percentile
mechanism for summarising vector-valued metrics.

The callback
------------

::

    def log_fn(metrics: dict[str, Any], steps: int) -> None: ...

``log_fn`` is called **once per training iteration** with:

- ``metrics`` — flat dict of metric name → scalar (or numpy / JAX
  array).
- ``steps`` — cumulative environment step count at the end of this
  iteration.

The dict is *flat* — nested metric trees are flattened with ``/``
between levels (e.g. ``"losses/predicted_value/mean"``).

Eval metrics are merged in **only on iterations where an eval ran**.
The idiom for "is this an eval iteration?" is the presence of an eval
key::

    def log_fn(metrics, steps):
        if "episode_reward_mean" in metrics:
            # an eval ran on this iteration — log the eval result
            ...

Logging levels
--------------

:class:`~nnx_ppo.algorithms.types.LoggingLevel` is a flag enum;
combine families with the ``|`` operator
(``LoggingLevel.LOSSES | LoggingLevel.GRAD_NORM``).

``LOSSES`` (alias ``BASIC``, the default)
    Always-on loss components: ``losses/actor_loss``,
    ``losses/critic_loss``, ``losses/entropy``, ``losses/total``, and
    the rollout's mean reward.
``CRITIC_EXTRA``
    Adds ``losses/predicted_value`` — the critic's value estimates
    for the rollout obs (mean/std or percentiles).
``ACTOR_EXTRA``
    Adds the rollout-action log-likelihoods, and (for 1-D actions) a
    Pearson correlation between log-likelihood and action.
``TRAIN_ROLLOUT_STATS``
    Adds ``rollout_batch/reward``, ``rollout_batch/action``,
    ``rollout_batch/done_rate``, ``rollout_batch/truncation_rate``.
``ROLLOUT_OBS``
    Reserved for future per-obs-key statistics.
``TRAINING_ENV_METRICS``
    Forwards the env's per-step ``state.metrics`` dict (e.g. survival
    bonuses, joint penalties — anything the env attaches to its
    state).
``GRAD_NORM``
    Adds ``grad_norm`` — global gradient norm after the gradient
    phase.
``WEIGHTS``
    Adds parameter statistics per layer (mean / std / percentiles of
    each ``nnx.Param``).
``THROUGHPUT``
    Adds wall-clock steps-per-second metrics (``throughput/train_sps``).
``ALL``
    Equivalent to ``LOSSES | CRITIC_EXTRA | ACTOR_EXTRA |
    TRAIN_ROLLOUT_STATS | TRAINING_ENV_METRICS | GRAD_NORM |
    WEIGHTS | ROLLOUT_OBS | THROUGHPUT``.
``NONE``
    No metrics. ``log_fn`` is still called with an empty dict.

Set the training level on :class:`PPOConfig.logging_level` and the
eval level on :class:`EvalConfig.logging_level`.

Percentile summaries
--------------------

Many of the metrics above are vector-valued (e.g. per-env rewards,
per-env action log-likelihoods). They are reduced to scalars before
being placed in the metrics dict. The reduction is controlled by the
``logging_percentiles`` field on both :class:`PPOConfig` and
:class:`EvalConfig`:

- ``logging_percentiles=None`` — emit ``"<name>/mean"`` and
  ``"<name>/std"`` keys.
- ``logging_percentiles=(0, 25, 50, 75, 100)`` — emit
  ``"<name>/p0"``, ``"<name>/p25"``, …, ``"<name>/p100"`` instead.

Pick whichever shape your downstream logging stack prefers.

Wiring to external loggers
--------------------------

``log_fn`` is just a callable, so any external logging library works
behind a thin user-side adapter::

    import wandb
    wandb.init(project="my-project")

    def log_fn(metrics, steps):
        wandb.log(metrics, step=steps)

    ppo.train_ppo(..., log_fn=log_fn)

For videos, a parallel ``video_fn`` callback receives a
:class:`~nnx_ppo.algorithms.config.VideoData` after each video
capture. Use :func:`~nnx_ppo.algorithms.callbacks.wandb_video_fn` to
plug straight into wandb::

    from nnx_ppo.algorithms.callbacks import wandb_video_fn

    ppo.train_ppo(..., log_fn=log_fn, video_fn=wandb_video_fn(fps=50))

Eval metric keys
----------------

Eval-iteration metric keys include (subject to ``EvalConfig.logging_level``
and ``EvalConfig.logging_percentiles``):

- ``episode_reward_mean`` / ``episode_reward_std`` (or
  ``episode_reward/p{N}`` if percentiles are configured),
- ``episode_length_mean`` / ``episode_length_std``,
- any env-side eval metrics surfaced when
  ``TRAINING_ENV_METRICS`` is in ``EvalConfig.logging_level``.
