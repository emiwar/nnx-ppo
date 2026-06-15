"""Tests for compute_metrics logging-level gating."""

from absl.testing import absltest
import jax.numpy as jp

from nnx_ppo.algorithms.metrics import compute_metrics
from nnx_ppo.algorithms.types import (
    LoggingLevel,
    PPONetworkOutput,
    Transition,
)


def _make_transition(T=2, B=3):
    obs = {"a": jp.ones((T, B, 4)), "b": {"c": jp.ones((T, B, 2))}}
    net_out = PPONetworkOutput(
        actions=jp.zeros((T, B, 2)),
        loglikelihoods=jp.zeros((T, B)),
        value_estimates=jp.zeros((T, B)),
    )
    return Transition(
        obs=obs,
        network_output=net_out,
        rewards=jp.ones((T, B)),
        done=jp.zeros((T, B), bool),
        truncated=jp.zeros((T, B), bool),
        next_obs=obs,
        metrics={
            "env": {"foo": jp.ones((T, B))},
            "net": {"bar": jp.ones((T, B)), "fm_pred_mse": jp.full((T, B), 2.0)},
        },
    )


class ComputeMetricsTest(absltest.TestCase):

    def _keys(self, logging_level):
        return set(
            compute_metrics({}, _make_transition(), logging_level, None).keys()
        )

    def test_env_metrics_only(self):
        keys = self._keys(LoggingLevel.ENV_METRICS)
        self.assertTrue(any(k.startswith("env/") for k in keys))
        self.assertFalse(any(k.startswith("net/") for k in keys))

    def test_network_metrics_only(self):
        keys = self._keys(LoggingLevel.NETWORK_METRICS)
        self.assertTrue(any(k.startswith("net/bar") for k in keys))
        self.assertTrue(any("fm_pred_mse" in k for k in keys))
        self.assertFalse(any(k.startswith("env/") for k in keys))

    def test_env_and_network_metrics_independent(self):
        keys = self._keys(LoggingLevel.ENV_METRICS | LoggingLevel.NETWORK_METRICS)
        self.assertTrue(any(k.startswith("env/") for k in keys))
        self.assertTrue(any(k.startswith("net/") for k in keys))

    def test_rollout_obs_logs_full_obs(self):
        keys = self._keys(LoggingLevel.ROLLOUT_OBS)
        self.assertTrue(any(k.startswith("rollout_batch/obs/a") for k in keys))
        self.assertTrue(any(k.startswith("rollout_batch/obs/b/c") for k in keys))
        # ROLLOUT_OBS must not drag in env/net metric subtrees.
        self.assertFalse(any(k.startswith("env/") or k.startswith("net/") for k in keys))

    def test_rollout_obs_off_by_default(self):
        self.assertNotIn(LoggingLevel.ROLLOUT_OBS, LoggingLevel.ALL)
        keys = self._keys(LoggingLevel.ALL)
        self.assertFalse(any(k.startswith("rollout_batch/obs") for k in keys))

    def test_rollout_stats(self):
        keys = self._keys(LoggingLevel.ROLLOUT_STATS)
        self.assertTrue(any(k.startswith("rollout_batch/reward") for k in keys))
        self.assertTrue(any(k.startswith("rollout_batch/action") for k in keys))
        self.assertIn("rollout_batch/done_rate", keys)
        self.assertIn("rollout_batch/truncation_rate", keys)


if __name__ == "__main__":
    absltest.main()
