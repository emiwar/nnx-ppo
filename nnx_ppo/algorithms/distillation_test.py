import os
import tempfile

from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx
import mujoco_playground

from nnx_ppo.networks import factories
from nnx_ppo.algorithms import distillation
from nnx_ppo.algorithms.checkpointing import load_checkpoint, make_checkpoint_fn
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.algorithms.config import DistillationConfig, DistillationTrainConfig, EvalConfig


def _make_env_and_nets(seed=17):
    env = mujoco_playground.registry.load(
        "CartpoleBalance", config_overrides={"impl": "jax"}
    )
    rngs_teacher = nnx.Rngs(seed, action_sampling=seed)
    rngs_student = nnx.Rngs(seed + 1, action_sampling=seed + 1)
    # Teacher and student must have isomorphic state trees so that the
    # teacher's rollout_extras can be fed directly to the student's
    # sampler during loss replay. Easiest way: same factory settings.
    teacher = factories.make_mlp_actor_critic(
        env.observation_size,
        env.action_size,
        actor_hidden_sizes=[16, 16],
        critic_hidden_sizes=[16, 16],
        rngs=rngs_teacher,
        normalize_obs=True,
    )
    student = factories.make_mlp_actor_critic(
        env.observation_size,
        env.action_size,
        actor_hidden_sizes=[16, 16],
        critic_hidden_sizes=[16, 16],
        rngs=rngs_student,
        normalize_obs=True,
    )
    return env, teacher, student


class DistillationStepTest(absltest.TestCase):

    def setUp(self):
        self.env, self.teacher, self.student = _make_env_and_nets()

    def test_distillation_step(self):
        config = DistillationConfig(n_envs=4, rollout_length=4, n_epochs=2, n_minibatches=2)
        state = distillation.new_distillation_state(
            self.env, self.teacher, self.student, config.n_envs, seed=18
        )
        self.assertEqual(int(state.steps_taken), 0)

        self.teacher.eval()
        state, metrics = distillation.distillation_step(
            self.env,
            self.teacher,
            state,
            config.n_envs,
            config.rollout_length,
            config.n_epochs,
            config.n_minibatches,
            LoggingLevel.ALL,
        )

        self.assertEqual(
            int(state.steps_taken), config.n_envs * config.rollout_length
        )
        for k, v in metrics.items():
            self.assertTrue(jp.all(jp.isfinite(v)), f"metrics[{k}] not finite")

    def test_distillation_step_twice(self):
        """Verify state continuity across consecutive steps."""
        config = DistillationConfig(n_envs=4, rollout_length=4, n_epochs=2, n_minibatches=2)
        state = distillation.new_distillation_state(
            self.env, self.teacher, self.student, config.n_envs, seed=19
        )
        self.teacher.eval()

        state, _ = distillation.distillation_step(
            self.env, self.teacher, state,
            config.n_envs, config.rollout_length, config.n_epochs, config.n_minibatches,
        )
        state, metrics = distillation.distillation_step(
            self.env, self.teacher, state,
            config.n_envs, config.rollout_length, config.n_epochs, config.n_minibatches,
        )

        self.assertEqual(
            int(state.steps_taken), config.n_envs * config.rollout_length * 2
        )

    def test_distillation_step_jit(self):
        config = DistillationConfig(n_envs=4, rollout_length=4, n_epochs=2, n_minibatches=2)
        state = distillation.new_distillation_state(
            self.env, self.teacher, self.student, config.n_envs, seed=20
        )
        self.teacher.eval()

        step_jit = nnx.jit(
            distillation.distillation_step, static_argnums=(0, 3, 4, 5, 6, 7, 8)
        )
        state, metrics = step_jit(
            self.env, self.teacher, state,
            config.n_envs, config.rollout_length, config.n_epochs, config.n_minibatches,
            LoggingLevel.LOSSES,
        )
        self.assertEqual(
            int(state.steps_taken), config.n_envs * config.rollout_length
        )
        for k, v in metrics.items():
            self.assertTrue(jp.all(jp.isfinite(v)), f"metrics[{k}] not finite")

        # Second call uses the cached JIT compilation.
        state, metrics = step_jit(
            self.env, self.teacher, state,
            config.n_envs, config.rollout_length, config.n_epochs, config.n_minibatches,
            LoggingLevel.LOSSES,
        )
        self.assertEqual(
            int(state.steps_taken), config.n_envs * config.rollout_length * 2
        )

    def test_teacher_params_unchanged(self):
        """Teacher parameters must not change across training iterations."""
        config = DistillationConfig(n_envs=4, rollout_length=4, n_epochs=2, n_minibatches=2)
        state = distillation.new_distillation_state(
            self.env, self.teacher, self.student, config.n_envs, seed=21
        )
        self.teacher.eval()

        teacher_params_before = jax.tree.map(
            lambda x: x.copy(), nnx.state(self.teacher, nnx.Param)
        )

        for _ in range(3):
            state, _ = distillation.distillation_step(
                self.env, self.teacher, state,
                config.n_envs, config.rollout_length, config.n_epochs, config.n_minibatches,
            )

        teacher_params_after = nnx.state(self.teacher, nnx.Param)

        jax.tree.map(
            lambda before, after: self.assertTrue(
                jp.allclose(before, after),
                "Teacher parameter changed during distillation!",
            ),
            teacher_params_before,
            teacher_params_after,
        )

    def test_distillation_loss_finite(self):
        """Unit test: distillation_loss returns finite values."""
        config = DistillationConfig(n_envs=4, rollout_length=4, n_epochs=2, n_minibatches=2)
        state = distillation.new_distillation_state(
            self.env, self.teacher, self.student, config.n_envs, seed=22
        )
        self.teacher.eval()

        # Get a rollout.
        _, _, _, rollout_data = distillation.distillation_unroll_env(
            self.env,
            state.env_states,
            self.teacher,
            self.student,
            state.student_states,
            state.teacher_states,
            config.rollout_length,
            jax.random.key(22),
        )

        minibatch_size = config.n_envs // config.n_minibatches
        minibatch_data = jax.tree.map(lambda x: x[:, :minibatch_size], rollout_data)
        student_state_subset = jax.tree.map(
            lambda x: x[:minibatch_size], state.student_states
        )

        loss, metrics = distillation.distillation_loss(
            self.student, student_state_subset, minibatch_data, LoggingLevel.LOSSES
        )
        self.assertTrue(jp.isfinite(loss), f"Loss not finite: {loss}")
        for k, v in metrics.items():
            self.assertTrue(jp.all(jp.isfinite(v)), f"metrics[{k}] not finite")


class TrainDistillationTest(absltest.TestCase):

    def test_train_distillation(self):
        env, teacher, student = _make_env_and_nets(seed=30)
        config = DistillationTrainConfig(
            distillation=DistillationConfig(
                n_envs=4, rollout_length=4, total_steps=64,
            ),
            eval=EvalConfig(enabled=False),
        )
        result = distillation.train_distillation(env, teacher, student, config)
        self.assertEqual(result.total_steps, 64)


# ---------------------------------------------------------------------------
# Non-isomorphic teacher / student, bridged by target_fn
# ---------------------------------------------------------------------------

def _unnormalized_teacher_target(teacher_extras, student_extras):
    """Map a bare-PPOAdapter teacher's extras onto a [Normalizer, PPOAdapter]
    student's: keep the student's Normalizer emission, take the adapter part
    (whose sampler leaf is the teacher mean) from the teacher."""
    return [student_extras[0], teacher_extras]


def _make_mismatched_nets(seed=40):
    """A teacher without a Normalizer and a student with one. Their
    rollout_extras trees differ by the leading Normalizer entry."""
    env = mujoco_playground.registry.load(
        "CartpoleBalance", config_overrides={"impl": "jax"}
    )
    kwargs = dict(actor_hidden_sizes=[16, 16], critic_hidden_sizes=[16, 16])
    teacher = factories.make_mlp_actor_critic(
        env.observation_size, env.action_size,
        rngs=nnx.Rngs(seed, action_sampling=seed), normalize_obs=False, **kwargs,
    )
    student = factories.make_mlp_actor_critic(
        env.observation_size, env.action_size,
        rngs=nnx.Rngs(seed + 1, action_sampling=seed + 1), normalize_obs=True,
        **kwargs,
    )
    return env, teacher, student


class TargetFnTest(absltest.TestCase):

    def test_target_is_student_shaped_and_holds_teacher_mean(self):
        env, teacher, student = _make_mismatched_nets()
        n_envs, T = 4, 4
        state = distillation.new_distillation_state(env, teacher, student, n_envs, seed=41)
        teacher.eval()

        _, _, _, rollout_data = distillation.distillation_unroll_env(
            env, state.env_states, teacher, student,
            state.student_states, state.teacher_states, T,
            jax.random.key(41), _unnormalized_teacher_target,
        )

        self.assertEqual(
            jax.tree.structure(rollout_data.teacher_rollout_extras),
            jax.tree.structure(rollout_data.student_rollout_extras),
        )
        # The teacher is a feedforward MLP in eval mode, so re-running it on the
        # stored observations reproduces its mean exactly.
        stored = rollout_data.teacher_rollout_extras[1]["action"][-1]
        for t in range(T):
            out = teacher(teacher.initialize_state(n_envs), rollout_data.obs[t])
            expected = out.rollout_extras["action"][-1]
            self.assertTrue(jp.allclose(stored[t], expected, atol=1e-5))

    def test_train_with_target_fn(self):
        env, teacher, student = _make_mismatched_nets(seed=42)
        config = DistillationTrainConfig(
            distillation=DistillationConfig(n_envs=4, rollout_length=4, total_steps=32),
            eval=EvalConfig(enabled=False),
        )
        result = distillation.train_distillation(
            env, teacher, student, config, target_fn=_unnormalized_teacher_target
        )
        self.assertEqual(result.total_steps, 32)


# ---------------------------------------------------------------------------
# Preemption support: stop_fn, initial_eval, checkpoints
# ---------------------------------------------------------------------------

class PreemptionTest(absltest.TestCase):

    def _config(self, total_steps=64, checkpoint_every_steps=1_000_000):
        return DistillationTrainConfig(
            distillation=DistillationConfig(
                n_envs=4, rollout_length=4, total_steps=total_steps,
            ),
            eval=EvalConfig(enabled=False),
            checkpoint_every_steps=checkpoint_every_steps,
        )

    def test_stop_fn_checkpoints_and_returns(self):
        env, teacher, student = _make_env_and_nets(seed=50)
        calls = []
        result = distillation.train_distillation(
            env, teacher, student, self._config(),
            checkpoint_fn=lambda state, step: calls.append(step),
            stop_fn=lambda steps: steps >= 32,
        )
        self.assertEqual(result.total_steps, 32)
        self.assertEqual(calls, [0, 32])

    def test_initial_eval_false_skips_step_zero(self):
        env, teacher, student = _make_env_and_nets(seed=51)
        calls = []
        distillation.train_distillation(
            env, teacher, student, self._config(checkpoint_every_steps=32),
            checkpoint_fn=lambda state, step: calls.append(step),
            initial_eval=False,
        )
        self.assertEqual(calls, [32, 64])

    def test_checkpoint_round_trip(self):
        env, teacher, student = _make_env_and_nets(seed=52)
        with tempfile.TemporaryDirectory() as tmp:
            result = distillation.train_distillation(
                env, teacher, student,
                self._config(total_steps=32, checkpoint_every_steps=32),
                checkpoint_fn=make_checkpoint_fn(tmp, include_env_state=False),
            )
            self.assertTrue(os.path.isdir(os.path.join(tmp, "step_0000000032")))

            _, _, fresh = _make_env_and_nets(seed=99)
            template = distillation.new_distillation_state(env, teacher, fresh, 1, seed=0)
            ckpt = load_checkpoint(
                os.path.join(tmp, "step_0000000032"), template.student, template.optimizer
            )

        self.assertEqual(int(ckpt["step"]), 32)
        self.assertEqual(int(ckpt["training_state"].steps_taken), 32)
        self.assertIsNone(ckpt["training_state"].network_states)
        trained = jax.tree.leaves(nnx.state(result.training_state.student, nnx.Param))
        restored = jax.tree.leaves(nnx.state(fresh, nnx.Param))
        for a, b in zip(trained, restored):
            self.assertTrue(jp.array_equal(a, b))


if __name__ == "__main__":
    absltest.main()
