"""Tests for non-finite diagnostics and the fatal-value check."""

import dataclasses

from absl.testing import absltest
from flax import nnx
import jax.numpy as jp

from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.metrics import (
    DIAGNOSTICS_PREFIX,
    check_diagnostics,
    compute_diagnostics,
    count_nonfinite,
)
from nnx_ppo.algorithms.types import (
    LoggingLevel,
    NonFiniteError,
    PPONetworkOutput,
    Transition,
)
from nnx_ppo.networks import factories
from nnx_ppo.test_dummies.move_to_center_env import MoveToCenterEnv


def _make_transition(T=2, B=3, reward=None, obs_val=None, next_obs_val=None,
                     action=None):
    obs = {"a": jp.full((T, B, 4), 1.0 if obs_val is None else obs_val)}
    next_obs = {"a": jp.full((T, B, 4), 1.0 if next_obs_val is None else next_obs_val)}
    net_out = PPONetworkOutput(
        actions=jp.full((T, B, 2), 0.0 if action is None else action),
        loglikelihoods=jp.zeros((T, B)),
        value_estimates=jp.zeros((T, B)),
    )
    return Transition(
        obs=obs,
        network_output=net_out,
        rewards=jp.full((T, B), 1.0 if reward is None else reward),
        done=jp.zeros((T, B), bool),
        truncated=jp.zeros((T, B), bool),
        next_obs=next_obs,
        metrics={"env": {}, "net": {}},
    )


def _diagnostics(transition, grad_nonfinite=0):
    return compute_diagnostics(
        transition, transition.network_output.actions, jp.array(grad_nonfinite)
    )


class CountNonFiniteTest(absltest.TestCase):

    def test_counts_nan_and_inf_across_leaves(self):
        tree = {"a": jp.array([jp.nan, 1.0]), "b": {"c": jp.array([jp.inf, -jp.inf, 2.0])}}
        self.assertEqual(int(count_nonfinite(tree)), 3)

    def test_clean_tree_is_zero(self):
        self.assertEqual(int(count_nonfinite({"a": jp.zeros((4, 5))})), 0)

    def test_ignores_non_float_leaves(self):
        tree = {"i": jp.arange(4), "b": jp.array([True, False]), "f": jp.array([jp.nan])}
        self.assertEqual(int(count_nonfinite(tree)), 1)

    def test_empty_tree_is_zero(self):
        self.assertEqual(int(count_nonfinite({})), 0)


class ComputeDiagnosticsTest(absltest.TestCase):

    def test_clean_rollout_reports_all_zero(self):
        d = _diagnostics(_make_transition())
        self.assertTrue(all(int(v) == 0 for v in d.values()))

    def test_localises_reward_obs_and_action(self):
        for field, key in [("reward", "nonfinite_reward"),
                           ("obs_val", "nonfinite_obs"),
                           ("action", "nonfinite_action"),
                           ("next_obs_val", "nonfinite_next_obs")]:
            d = _diagnostics(_make_transition(**{field: jp.nan}))
            hit = [k for k, v in d.items() if int(v) > 0]
            self.assertEqual(hit, [DIAGNOSTICS_PREFIX + key], msg=field)


class CheckDiagnosticsTest(absltest.TestCase):

    def test_clean_metrics_pass(self):
        check_diagnostics(_diagnostics(_make_transition()), 7)

    def test_raises_on_nonfinite_reward(self):
        m = _diagnostics(_make_transition(reward=jp.nan))
        with self.assertRaises(NonFiniteError) as cm:
            check_diagnostics(m, 1234)
        self.assertEqual(cm.exception.step, 1234)
        self.assertEqual(cm.exception.counts[DIAGNOSTICS_PREFIX + "nonfinite_reward"], 6)

    def test_nonfinite_next_obs_alone_is_not_fatal(self):
        """An env flagging its own divergence: the reset in `unroll_env` and
        GAE's `where(done, 0, next_value)` keep it out of every gradient."""
        m = _diagnostics(_make_transition(next_obs_val=jp.nan))
        check_diagnostics(m, 0)

    def test_reports_all_counts_including_non_fatal(self):
        m = _diagnostics(_make_transition(reward=jp.nan, next_obs_val=jp.nan))
        with self.assertRaises(NonFiniteError) as cm:
            check_diagnostics(m, 0)
        self.assertGreater(
            cm.exception.counts[DIAGNOSTICS_PREFIX + "nonfinite_next_obs"], 0
        )

    def test_missing_diagnostics_are_tolerated(self):
        check_diagnostics({"losses/actor/mean": 0.5}, 0)


def _nets(seed=0):
    return factories.make_mlp_actor_critic(
        rngs=nnx.Rngs(seed), obs_size=2, action_size=2,
        actor_hidden_sizes=[8, 8], critic_hidden_sizes=[8, 8],
        normalize_obs=True, entropy_weight=0.01)


def _cfg(logging_level, n_iters=2):
    c = ppo.default_config()
    return dataclasses.replace(c, ppo=dataclasses.replace(
        c.ppo, n_envs=8, rollout_length=4, total_steps=8 * 4 * n_iters,
        n_epochs=1, n_minibatches=2, logging_level=logging_level),
        eval=dataclasses.replace(c.eval, enabled=False),
        video=dataclasses.replace(c.video, enabled=False))


class _PoisonRewardEnv(MoveToCenterEnv):
    """Emits a NaN reward from the first step: a stand-in for a divergence
    whose NaN also reaches the fields the reward function reads."""

    def step(self, state, action):
        s = super().step(state, action)
        return s.replace(reward=jp.full_like(s.reward, jp.nan))


class TrainPPODiagnosticsTest(absltest.TestCase):

    def _run(self, env, logging_level, n_iters=2):
        logged = []
        ppo.train_ppo(env, _nets(), _cfg(logging_level, n_iters),
                      log_fn=lambda m, step: logged.append(dict(m)))
        return logged

    def test_diagnostics_logged_when_enabled(self):
        logged = self._run(MoveToCenterEnv(),
                           LoggingLevel.LOSSES | LoggingLevel.DIAGNOSTICS)
        keys = {k for k in logged[-1] if k.startswith(DIAGNOSTICS_PREFIX)}
        self.assertIn(DIAGNOSTICS_PREFIX + "nonfinite_reward", keys)
        self.assertIn(DIAGNOSTICS_PREFIX + "nonfinite_grad", keys)

    def test_diagnostics_absent_from_metrics_when_disabled(self):
        logged = self._run(MoveToCenterEnv(), LoggingLevel.LOSSES)
        for row in logged:
            self.assertEqual(
                [k for k in row if k.startswith(DIAGNOSTICS_PREFIX)], []
            )

    def test_check_runs_even_when_diagnostics_not_logged(self):
        """The check is a safety net, not a logging feature: gating it on the
        LoggingLevel would make the guard opt-in."""
        with self.assertRaises(NonFiniteError):
            self._run(_PoisonRewardEnv(), LoggingLevel.LOSSES)

    def test_error_localises_the_env_as_the_source(self):
        with self.assertRaises(NonFiniteError) as cm:
            self._run(_PoisonRewardEnv(), LoggingLevel.LOSSES)
        counts = cm.exception.counts
        self.assertGreater(counts[DIAGNOSTICS_PREFIX + "nonfinite_reward"], 0)
        self.assertEqual(counts[DIAGNOSTICS_PREFIX + "nonfinite_obs"], 0)
        self.assertEqual(counts[DIAGNOSTICS_PREFIX + "nonfinite_action"], 0)
        self.assertGreater(counts[DIAGNOSTICS_PREFIX + "nonfinite_grad"], 0)

    def test_no_checkpoint_written_for_the_corrupt_update(self):
        saved = []
        with self.assertRaises(NonFiniteError):
            ppo.train_ppo(
                _PoisonRewardEnv(), _nets(), _cfg(LoggingLevel.LOSSES, n_iters=3),
                checkpoint_fn=lambda training_state, step: saved.append(step),
            )
        self.assertEqual(saved, [0])


class _NaNRewardCartpole:
    """CartpoleBalance with a NaN reward, to exercise the distillation guard."""

    def __init__(self):
        import mujoco_playground
        self._env = mujoco_playground.registry.load(
            "CartpoleBalance", config_overrides={"impl": "jax"}
        )

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, rng):
        return self._env.reset(rng)

    def step(self, state, action):
        s = self._env.step(state, action)
        return s.replace(reward=jp.full_like(s.reward, jp.nan))


class TrainDistillationDiagnosticsTest(absltest.TestCase):
    """The distillation loop has its own scan and its own metrics dict, so the
    guard is wired separately there and needs its own coverage."""

    def _setup(self, seed=30):
        import mujoco_playground
        from nnx_ppo.algorithms.config import (
            DistillationConfig,
            DistillationTrainConfig,
            EvalConfig,
        )
        env = mujoco_playground.registry.load(
            "CartpoleBalance", config_overrides={"impl": "jax"}
        )
        def net(s):
            return factories.make_mlp_actor_critic(
                env.observation_size, env.action_size,
                actor_hidden_sizes=[8, 8], critic_hidden_sizes=[8, 8],
                rngs=nnx.Rngs(s, action_sampling=s), normalize_obs=True)
        return env, net(seed), net(seed + 1), DistillationConfig, \
            DistillationTrainConfig, EvalConfig

    def test_diagnostics_logged_when_enabled(self):
        from nnx_ppo.algorithms import distillation
        env, teacher, student, DC, DTC, EC = self._setup()
        logged = []
        distillation.train_distillation(
            env, teacher, student,
            DTC(distillation=DC(n_envs=4, rollout_length=4, total_steps=32,
                                logging_level=LoggingLevel.LOSSES
                                | LoggingLevel.DIAGNOSTICS),
                eval=EC(enabled=False)),
            log_fn=lambda m, step: logged.append(dict(m)))
        keys = {k for k in logged[-1] if k.startswith(DIAGNOSTICS_PREFIX)}
        self.assertIn(DIAGNOSTICS_PREFIX + "nonfinite_grad", keys)
        self.assertIn(DIAGNOSTICS_PREFIX + "nonfinite_reward", keys)

    def test_diagnostics_absent_when_disabled(self):
        from nnx_ppo.algorithms import distillation
        env, teacher, student, DC, DTC, EC = self._setup()
        logged = []
        distillation.train_distillation(
            env, teacher, student,
            DTC(distillation=DC(n_envs=4, rollout_length=4, total_steps=32,
                                logging_level=LoggingLevel.LOSSES),
                eval=EC(enabled=False)),
            log_fn=lambda m, step: logged.append(dict(m)))
        for row in logged:
            self.assertEqual(
                [k for k in row if k.startswith(DIAGNOSTICS_PREFIX)], []
            )

    def test_raises_on_nonfinite_reward(self):
        from nnx_ppo.algorithms import distillation
        _, teacher, student, DC, DTC, EC = self._setup(seed=40)
        with self.assertRaises(NonFiniteError):
            distillation.train_distillation(
                _NaNRewardCartpole(), teacher, student,
                DTC(distillation=DC(n_envs=4, rollout_length=4, total_steps=32,
                                    logging_level=LoggingLevel.LOSSES),
                    eval=EC(enabled=False)))


if __name__ == "__main__":
    absltest.main()
