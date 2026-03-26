from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx
import mujoco_playground

from nnx_ppo.networks import factories
from nnx_ppo.algorithms import distillation
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.algorithms.config import DistillationConfig, DistillationTrainConfig, EvalConfig


def _make_env_and_nets(seed=17):
    env = mujoco_playground.registry.load("CartpoleBalance")
    rngs_teacher = nnx.Rngs(seed, action_sampling=seed)
    rngs_student = nnx.Rngs(seed + 1, action_sampling=seed + 1)
    teacher = factories.make_mlp_actor_critic(
        env.observation_size,
        env.action_size,
        actor_hidden_sizes=[16, 16],
        critic_hidden_sizes=[16, 16],
        rngs=rngs_teacher,
        normalize_obs=False,
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


if __name__ == "__main__":
    absltest.main()
