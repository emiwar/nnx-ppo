"""Metrics computation and logging utilities for PPO training."""

from typing import Any, Optional, Union
from collections.abc import Mapping
import warnings

import jax
import jax.numpy as jp
from flax import nnx
from jaxtyping import Array, Float, PyTree

from nnx_ppo.networks.types import StatefulModule
from nnx_ppo.algorithms.types import LoggingLevel, NonFiniteError
from nnx_ppo.algorithms.rollout import Transition


def compute_metrics(
    loss_metrics: dict[str, PyTree],
    rollout_data: Transition,
    logging_level: LoggingLevel,
    percentile_levels: Optional[tuple[int, ...]] = None,
) -> dict[str, Any]:
    """Compute training metrics from loss metrics and rollout data.

    Args:
        loss_metrics: Dictionary of loss values from ppo_loss.
        rollout_data: Transition data from the rollout.
        logging_level: Which metrics to include.
        percentile_levels: Percentiles to compute (e.g., (0, 25, 50, 75, 100)).
                          If None, uses mean/std instead.

    Returns:
        Dictionary of computed metrics.
    """
    metrics = {}
    for k, v in loss_metrics.items():
        _log_metric(metrics, k, v, percentile_levels)
    if LoggingLevel.ENV_METRICS in logging_level:
        _log_metric(metrics, "env", rollout_data.metrics["env"], percentile_levels)
    if LoggingLevel.NETWORK_METRICS in logging_level:
        _log_metric(metrics, "net", rollout_data.metrics["net"], percentile_levels)
    if LoggingLevel.ROLLOUT_STATS in logging_level:
        _log_metric(
            metrics, "rollout_batch/reward", rollout_data.rewards, percentile_levels
        )
        _log_metric(
            metrics,
            "rollout_batch/action",
            rollout_data.network_output.actions,
            percentile_levels,
        )
        metrics["rollout_batch/done_rate"] = rollout_data.done.mean()
        metrics["rollout_batch/truncation_rate"] = rollout_data.truncated.mean()
    if LoggingLevel.ROLLOUT_OBS in logging_level:
        _log_metric(
            metrics, "rollout_batch/obs", rollout_data.obs, percentile_levels
        )
    if LoggingLevel.ACTOR_EXTRA in logging_level:
        _log_metric(
            metrics,
            "loglikelihood",
            rollout_data.network_output.loglikelihoods,
            percentile_levels,
        )
    if LoggingLevel.CRITIC_EXTRA in logging_level:
        _log_metric(
            metrics,
            "losses/predicted_value",
            rollout_data.network_output.value_estimates,
            percentile_levels,
        )
    return metrics


#: Prefix for every key :func:`compute_diagnostics` produces.
DIAGNOSTICS_PREFIX = "diagnostics/"

#: Diagnostics that mean the update is already corrupt. A non-finite reward,
#: observation, action or gradient all reach the batch-mean reductions in
#: `ppo_loss`, so one of them anywhere turns *every* gradient non-finite, and
#: Adam's moments then make it permanent.
FATAL_DIAGNOSTICS = (
    f"{DIAGNOSTICS_PREFIX}nonfinite_reward",
    f"{DIAGNOSTICS_PREFIX}nonfinite_obs",
    f"{DIAGNOSTICS_PREFIX}nonfinite_action",
    f"{DIAGNOSTICS_PREFIX}nonfinite_grad",
)


def count_nonfinite(tree: Any) -> Array:
    """Number of NaN/Inf elements across a pytree's floating-point leaves."""
    leaves = [
        x for x in jax.tree.leaves(tree)
        if hasattr(x, "dtype") and jp.issubdtype(x.dtype, jp.inexact)
    ]
    if not leaves:
        return jp.array(0, jp.int32)
    return sum(jp.sum(~jp.isfinite(x)).astype(jp.int32) for x in leaves)


def compute_diagnostics(
    rollout_data: Union[Transition, Any],
    actions: Any,
    grad_nonfinite: Array,
) -> dict[str, Array]:
    """Non-finite element counts for one iteration, as int scalars.

    ``actions`` is passed separately rather than read off ``rollout_data`` so
    that this serves both `Transition` (``network_output``) and
    `DistillationTransition` (``student_output``).

    ``next_obs`` is reported but is *not* fatal: an env that flags its own
    divergence (``done=1``) has its next state replaced by a reset in
    `unroll_env`, and GAE drops the bootstrap through
    ``jp.where(done, 0.0, next_value)``, so a non-finite ``next_obs`` on a
    terminated step never reaches a gradient.
    """
    return {
        f"{DIAGNOSTICS_PREFIX}nonfinite_reward": count_nonfinite(rollout_data.rewards),
        f"{DIAGNOSTICS_PREFIX}nonfinite_obs": count_nonfinite(rollout_data.obs),
        f"{DIAGNOSTICS_PREFIX}nonfinite_action": count_nonfinite(actions),
        f"{DIAGNOSTICS_PREFIX}nonfinite_grad": jp.asarray(grad_nonfinite, jp.int32),
        f"{DIAGNOSTICS_PREFIX}nonfinite_next_obs": count_nonfinite(
            rollout_data.next_obs
        ),
    }


def check_diagnostics(metrics: Mapping[str, Any], step: int) -> None:
    """Raise :class:`NonFiniteError` if this iteration produced a fatal non-finite.

    Host-side, so it forces a device sync — but the training loop already syncs
    on ``steps_taken`` once per iteration, so reading these scalars in the same
    barrier costs nothing measurable.
    """
    counts = {
        k: int(metrics[k]) for k in FATAL_DIAGNOSTICS if k in metrics
    }
    hit = {k: v for k, v in counts.items() if v > 0}
    if not hit:
        return
    reported = dict(counts)
    for k, v in metrics.items():
        if k.startswith(DIAGNOSTICS_PREFIX) and k not in reported:
            reported[k] = int(v)
    where = ", ".join(
        f"{k[len(DIAGNOSTICS_PREFIX):]}={v}" for k, v in sorted(hit.items())
    )
    raise NonFiniteError(
        f"non-finite values at step {step}: {where}. "
        f"All counts: {reported}. Training stopped before the corrupt update was "
        f"checkpointed; the previous checkpoint is unaffected. A single non-finite "
        f"reward or observation anywhere in the batch turns every gradient non-finite, "
        f"so this is not recoverable by continuing.",
        counts=reported,
        step=step,
    )


def _log_metric(
    metrics: dict[str, Any],
    name: str,
    x: Union[Mapping[Union[str, int], Any], Float[Array, "..."]],
    percentile_levels: Optional[tuple[int, ...]] = None,
) -> None:
    """Log a metric with either percentiles or mean/std.

    Args:
        metrics: Dictionary to add metrics to (mutated in place).
        name: Base name for the metric.
        x: Value to log (can be array or nested mapping).
        percentile_levels: Percentiles to compute. If None, uses mean/std.
    """
    if isinstance(x, Mapping):
        for k, v in x.items():
            _log_metric(metrics, f"{name}/{k}", v, percentile_levels)
        return

    # Boolean arrays (e.g. termination flags): log fraction-true, not mean/std/percentiles
    if hasattr(x, "dtype") and jp.issubdtype(x.dtype, jp.bool_):
        metrics[name] = jp.mean(x)
    elif percentile_levels is None or len(percentile_levels) == 0:
        metrics[f"{name}/mean"] = jp.mean(x)
        metrics[f"{name}/std"] = jp.std(x)
    else:
        percentiles = jp.percentile(x, jp.array(percentile_levels))
        for pl, p in zip(percentile_levels, percentiles):
            metrics[f"{name}/p{int(pl)}"] = p


def log_weight_stats(
    metrics: dict[str, Any],
    networks: StatefulModule,
    percentile_levels: Optional[tuple[int, ...]] = None,
) -> None:
    """Log weight statistics over all of the network's parameters.

    Args:
        metrics: Dictionary to add metrics to (mutated in place).
        networks: The PPO network to extract weights from.
        percentile_levels: Percentiles to compute. If None, uses mean/std.
    """
    params = nnx.state(networks, nnx.Param)
    leaves = jax.tree.leaves(params)
    if not leaves:
        warnings.warn("Network has no nnx.Param leaves; skipping weight logging.")
        return None
    weights = jp.concatenate([p.flatten() for p in leaves])
    _log_metric(metrics, "weights", weights, percentile_levels)
