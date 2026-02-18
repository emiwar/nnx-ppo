"""Metrics computation and logging utilities for PPO training."""
from typing import Any, Dict, Optional, Tuple, Union, Mapping

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.types import PPONetwork
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.algorithms.rollout import Transition


def compute_metrics(loss_metrics: Dict[str, jax.Array],
                    rollout_data: Transition,
                    logging_level: LoggingLevel,
                    percentile_levels: Optional[Tuple] = None) -> Dict[str, Any]:
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
    if LoggingLevel.TRAINING_ENV_METRICS in logging_level:
        for k, v in rollout_data.metrics.items():
            _log_metric(metrics, k, v, percentile_levels)
    if LoggingLevel.TRAIN_ROLLOUT_STATS in logging_level:
        _log_metric(metrics, "rollout_batch/reward", rollout_data.rewards, percentile_levels)
        _log_metric(metrics, "rollout_batch/action", rollout_data.network_output.actions, percentile_levels)
        metrics["rollout_batch/done_rate"] = rollout_data.done.mean()
        metrics["rollout_batch/truncation_rate"] = rollout_data.truncated.mean()
    if LoggingLevel.ROLLOUT_OBS in logging_level:
        pass
    if LoggingLevel.ACTOR_EXTRA in logging_level:
        _log_metric(metrics, "loglikelihood", rollout_data.network_output.loglikelihoods, percentile_levels)
        if rollout_data.network_output.actions.shape[-1] == 1:
            metrics["correlations/action_ll"] = jp.corrcoef(rollout_data.network_output.loglikelihoods.flatten(),
                                                    rollout_data.network_output.actions.flatten())[0, 1]
    if LoggingLevel.CRITIC_EXTRA in logging_level:
        _log_metric(metrics, "losses/predicted_value", rollout_data.network_output.value_estimates, percentile_levels)
    return metrics


def _log_metric(metrics: Dict[str, jax.Array],
                name: str,
                x: Union[Mapping, jax.Array],
                percentile_levels: Optional[Tuple] = None) -> None:
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
    if name.startswith("env/termination"):  # These are boolean, but casted to float earlier
        metrics[name] = jp.mean(x)
    elif percentile_levels is None or len(percentile_levels) == 0:
        metrics[f"{name}/mean"] = jp.mean(x)
        metrics[f"{name}/std"] = jp.std(x)
    else:
        percentiles = jp.percentile(x, jp.array(percentile_levels))
        for (pl, p) in zip(percentile_levels, percentiles):
            metrics[f"{name}/p{int(pl)}"] = p


def log_weight_stats(metrics: Dict[str, jax.Array],
                     networks: PPONetwork,
                     percentile_levels: Optional[Tuple] = None) -> None:
    """Log weight statistics for actor and critic networks separately.

    Args:
        metrics: Dictionary to add metrics to (mutated in place).
        networks: The PPO network to extract weights from.
        percentile_levels: Percentiles to compute. If None, uses mean/std.
    """
    # Extract parameters using nnx.state
    actor_params = nnx.state(networks.actor, nnx.Param)
    critic_params = nnx.state(networks.critic, nnx.Param)

    # Flatten all actor weights into single array
    actor_weights = jp.concatenate([p.flatten() for p in jax.tree.leaves(actor_params)])
    critic_weights = jp.concatenate([p.flatten() for p in jax.tree.leaves(critic_params)])

    _log_metric(metrics, "weights/actor", actor_weights, percentile_levels)
    _log_metric(metrics, "weights/critic", critic_weights, percentile_levels)
