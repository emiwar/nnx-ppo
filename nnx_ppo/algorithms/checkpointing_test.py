"""Tests for checkpointing utilities."""

import os
import shutil
import tempfile

from absl.testing import absltest
import jax.numpy as jp
import mujoco_playground
from flax import nnx

from nnx_ppo.networks import factories
from nnx_ppo.networks.containers import PPOActorCritic, Sequential
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.variational import AR1VariationalBottleneck
from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.checkpointing import make_checkpoint_fn, load_checkpoint
from nnx_ppo.algorithms.config import PPOConfig, EvalConfig, TrainConfig
from nnx_ppo.algorithms.types import TrainingState


def _make_nets(env, seed: int = 17):
    return factories.make_mlp_actor_critic(
        env.observation_size,
        env.action_size,
        actor_hidden_sizes=[8, 8],
        critic_hidden_sizes=[8, 8],
        rngs=nnx.Rngs(seed),
        normalize_obs=False,
    )


def _make_norm_nets(env, seed: int = 17):
    """Networks with observation normalization (has a Normalizer preprocessor)."""
    return factories.make_mlp_actor_critic(
        env.observation_size,
        env.action_size,
        actor_hidden_sizes=[8, 8],
        critic_hidden_sizes=[8, 8],
        rngs=nnx.Rngs(seed),
        normalize_obs=True,
    )


_LATENT_SIZE = 4


def _make_ar1vb_nets(env, seed: int = 17):
    """Networks with AR1VariationalBottleneck in the actor."""
    rngs = nnx.Rngs(seed)
    obs_size = env.observation_size
    action_size = env.action_size

    actor = Sequential([
        Dense(obs_size, _LATENT_SIZE * 2, rngs, activation=nnx.relu),
        AR1VariationalBottleneck(_LATENT_SIZE, rng=rngs.params),
        Dense(_LATENT_SIZE, action_size * 2, rngs),
    ])
    critic = Sequential([
        Dense(obs_size, 8, rngs, activation=nnx.relu),
        Dense(8, 1, rngs),
    ])
    action_sampler = NormalTanhSampler(rngs, entropy_weight=1e-2)
    return PPOActorCritic(actor=actor, critic=critic, action_sampler=action_sampler)


class TrainConfigCheckpointTest(absltest.TestCase):

    def test_default_checkpoint_every_steps(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.checkpoint_every_steps, 500_000)

    def test_custom_checkpoint_every_steps(self):
        cfg = TrainConfig(checkpoint_every_steps=100_000)
        self.assertEqual(cfg.checkpoint_every_steps, 100_000)


class MakeCheckpointFnTest(absltest.TestCase):

    def setUp(self):
        self.env = mujoco_playground.registry.load("CartpoleBalance")
        self.nets = _make_nets(self.env, seed=17)

    def _make_training_state(self, n_envs: int = 4, seed: int = 42) -> TrainingState:
        return ppo.new_training_state(self.env, self.nets, n_envs, seed)

    def _fresh_state(self, n_envs: int = 4, seed: int = 99) -> TrainingState:
        """Create a fresh TrainingState with same architecture (different weights)."""
        fresh_nets = _make_nets(self.env, seed=seed)
        return ppo.new_training_state(self.env, fresh_nets, n_envs, seed)

    def test_creates_step_directory(self):
        tmpdir = tempfile.mkdtemp()
        try:
            state = self._make_training_state()
            fn = make_checkpoint_fn(tmpdir)
            fn(state, step=5000)
            expected = os.path.join(tmpdir, "step_0000005000")
            self.assertTrue(os.path.isdir(expected))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_creates_base_directory_if_missing(self):
        tmpdir = tempfile.mkdtemp()
        try:
            base = os.path.join(tmpdir, "new_subdir")
            self.assertFalse(os.path.exists(base))
            state = self._make_training_state()
            fn = make_checkpoint_fn(base)
            fn(state, step=1)
            self.assertTrue(os.path.isdir(base))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_saves_expected_contents(self):
        tmpdir = tempfile.mkdtemp()
        try:
            state = self._make_training_state()
            fn = make_checkpoint_fn(tmpdir)
            fn(state, step=1000)
            step_dir = os.path.join(tmpdir, "step_0000001000")
            self.assertTrue(os.path.isdir(os.path.join(step_dir, "networks")))
            self.assertTrue(os.path.isdir(os.path.join(step_dir, "optimizer")))
            self.assertTrue(os.path.isfile(os.path.join(step_dir, "metadata.pkl")))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_round_trip_without_config(self):
        tmpdir = tempfile.mkdtemp()
        try:
            state = self._make_training_state()
            fn = make_checkpoint_fn(tmpdir)
            fn(state, step=1234)
            template = self._fresh_state()
            ckpt = load_checkpoint(
                os.path.join(tmpdir, "step_0000001234"),
                template.networks,
                template.optimizer,
            )
            self.assertIn("training_state", ckpt)
            self.assertIn("step", ckpt)
            self.assertIn("config", ckpt)
            self.assertEqual(ckpt["step"], 1234)
            self.assertIsNone(ckpt["config"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_round_trip_with_config(self):
        tmpdir = tempfile.mkdtemp()
        try:
            config = TrainConfig(
                ppo=PPOConfig(n_envs=4, total_steps=1000),
                eval=EvalConfig(enabled=False),
            )
            state = self._make_training_state()
            fn = make_checkpoint_fn(tmpdir, config=config)
            fn(state, step=1000)
            template = self._fresh_state()
            ckpt = load_checkpoint(
                os.path.join(tmpdir, "step_0000001000"),
                template.networks,
                template.optimizer,
            )
            self.assertIsNotNone(ckpt["config"])
            self.assertEqual(ckpt["config"].ppo.n_envs, 4)
            self.assertEqual(ckpt["config"].ppo.total_steps, 1000)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_steps_taken_preserved(self):
        tmpdir = tempfile.mkdtemp()
        try:
            state = self._make_training_state()
            state = state.replace(steps_taken=jp.array(9999.0))
            fn = make_checkpoint_fn(tmpdir)
            fn(state, step=9999)
            template = self._fresh_state()
            ckpt = load_checkpoint(
                os.path.join(tmpdir, "step_0000009999"),
                template.networks,
                template.optimizer,
            )
            loaded = ckpt["training_state"]
            self.assertIsInstance(loaded, TrainingState)
            self.assertAlmostEqual(float(loaded.steps_taken), 9999.0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_network_weights_preserved(self):
        tmpdir = tempfile.mkdtemp()
        try:
            state = self._make_training_state()
            fn = make_checkpoint_fn(tmpdir)
            fn(state, step=500)

            # Load into a fresh template with different random weights.
            template = self._fresh_state()
            ckpt = load_checkpoint(
                os.path.join(tmpdir, "step_0000000500"),
                template.networks,
                template.optimizer,
            )
            loaded = ckpt["training_state"]

            # Run a forward pass with original and loaded networks; outputs must match.
            obs = jp.zeros((4, self.env.observation_size))
            _, out_orig = state.networks(state.network_states, obs)
            _, out_loaded = loaded.networks(loaded.network_states, obs)
            self.assertTrue(
                jp.allclose(out_orig.value_estimates, out_loaded.value_estimates),
                "Value estimates differ after checkpoint round-trip.",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_multiple_checkpoints_accumulate(self):
        tmpdir = tempfile.mkdtemp()
        try:
            state = self._make_training_state()
            fn = make_checkpoint_fn(tmpdir)
            fn(state, step=1000)
            fn(state, step=2000)
            fn(state, step=3000)
            dirs = sorted(
                d
                for d in os.listdir(tmpdir)
                if os.path.isdir(os.path.join(tmpdir, d))
            )
            self.assertEqual(len(dirs), 3)
            self.assertIn("step_0000001000", dirs)
            self.assertIn("step_0000002000", dirs)
            self.assertIn("step_0000003000", dirs)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class NormalizerCheckpointTest(absltest.TestCase):
    """Verify that NormalizerStatistics (a custom nnx.Variable subclass) are
    correctly saved and restored.  These variables are NOT nnx.Param, so an
    earlier (param-only) split strategy would silently drop them."""

    def setUp(self):
        self.env = mujoco_playground.registry.load("CartpoleBalance")
        self.nets = _make_norm_nets(self.env, seed=17)

    def _make_training_state(self, n_envs: int = 4, seed: int = 42) -> TrainingState:
        return ppo.new_training_state(self.env, self.nets, n_envs, seed)

    def _fresh_state(self, n_envs: int = 4, seed: int = 99) -> TrainingState:
        fresh_nets = _make_norm_nets(self.env, seed=seed)
        return ppo.new_training_state(self.env, fresh_nets, n_envs, seed)

    def test_normalizer_statistics_preserved(self):
        """NormalizerStatistics variables survive a checkpoint round-trip."""
        tmpdir = tempfile.mkdtemp()
        try:
            state = self._make_training_state()

            # Set known, non-zero normalizer statistics.
            obs_size = self.env.observation_size
            state.networks.preprocessor.mean[...] = jp.ones(obs_size) * 3.14
            state.networks.preprocessor.M2[...] = jp.ones(obs_size) * 2.0
            state.networks.preprocessor.counter[...] = jp.array(1000.0)

            fn = make_checkpoint_fn(tmpdir)
            fn(state, step=100)

            template = self._fresh_state()
            ckpt = load_checkpoint(
                os.path.join(tmpdir, "step_0000000100"),
                template.networks,
                template.optimizer,
            )
            loaded = ckpt["training_state"]

            self.assertTrue(
                jp.allclose(
                    loaded.networks.preprocessor.mean[...],
                    state.networks.preprocessor.mean[...],
                ),
                "Normalizer mean not preserved after checkpoint.",
            )
            self.assertTrue(
                jp.allclose(
                    loaded.networks.preprocessor.M2[...],
                    state.networks.preprocessor.M2[...],
                ),
                "Normalizer M2 not preserved after checkpoint.",
            )
            self.assertAlmostEqual(
                float(loaded.networks.preprocessor.counter[...]),
                1000.0,
                msg="Normalizer counter not preserved after checkpoint.",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class AR1VBCheckpointTest(absltest.TestCase):
    """Verify that AR1VariationalBottleneck state (last_z, PRNG keys) stored in
    network_states is correctly preserved across a checkpoint round-trip."""

    def setUp(self):
        self.env = mujoco_playground.registry.load("CartpoleBalance")
        self.nets = _make_ar1vb_nets(self.env, seed=17)

    def _make_training_state(self, n_envs: int = 4, seed: int = 42) -> TrainingState:
        return ppo.new_training_state(self.env, self.nets, n_envs, seed)

    def _fresh_state(self, n_envs: int = 4, seed: int = 99) -> TrainingState:
        fresh_nets = _make_ar1vb_nets(self.env, seed=seed)
        return ppo.new_training_state(self.env, fresh_nets, n_envs, seed)

    def test_ar1vb_last_z_preserved(self):
        """AR1VB last_z in network_states survives a checkpoint round-trip."""
        n_envs = 4
        tmpdir = tempfile.mkdtemp()
        try:
            state = self._make_training_state(n_envs=n_envs)

            # AR1VB is layer index 1 in the actor Sequential: [Dense, AR1VB, Dense].
            actor_state = state.network_states["actor"]
            known_last_z = jp.ones((n_envs, _LATENT_SIZE)) * 7.77
            new_ar1vb_state = {"keys": actor_state[1]["keys"], "last_z": known_last_z}
            new_network_states = dict(state.network_states)
            new_network_states["actor"] = [actor_state[0], new_ar1vb_state, actor_state[2]]
            state = state.replace(network_states=new_network_states)

            fn = make_checkpoint_fn(tmpdir)
            fn(state, step=200)

            template = self._fresh_state(n_envs=n_envs)
            ckpt = load_checkpoint(
                os.path.join(tmpdir, "step_0000000200"),
                template.networks,
                template.optimizer,
            )
            loaded = ckpt["training_state"]

            loaded_last_z = loaded.network_states["actor"][1]["last_z"]
            self.assertTrue(
                jp.allclose(loaded_last_z, known_last_z),
                "AR1VB last_z not preserved after checkpoint.",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_ar1vb_nan_sentinel_preserved(self):
        """AR1VB last_z initialized to NaN (reset sentinel) survives round-trip."""
        n_envs = 4
        tmpdir = tempfile.mkdtemp()
        try:
            state = self._make_training_state(n_envs=n_envs)
            # Initial last_z should be all NaN (from initialize_state).
            initial_last_z = state.network_states["actor"][1]["last_z"]
            self.assertTrue(
                jp.all(jp.isnan(initial_last_z)),
                "Expected initial last_z to be all NaN.",
            )

            fn = make_checkpoint_fn(tmpdir)
            fn(state, step=0)

            template = self._fresh_state(n_envs=n_envs)
            ckpt = load_checkpoint(
                os.path.join(tmpdir, "step_0000000000"),
                template.networks,
                template.optimizer,
            )
            loaded = ckpt["training_state"]

            loaded_last_z = loaded.network_states["actor"][1]["last_z"]
            self.assertTrue(
                jp.all(jp.isnan(loaded_last_z)),
                "AR1VB NaN sentinel not preserved after checkpoint.",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class CheckpointFnInvocationTest(absltest.TestCase):
    """Test checkpoint_fn is invoked at the correct intervals by train_ppo.

    These tests trigger JIT compilation and are therefore slow.
    """

    def setUp(self):
        self.env = mujoco_playground.registry.load("CartpoleBalance")
        self.nets = _make_nets(self.env, seed=17)

    def _tiny_config(self, total_steps: int, checkpoint_every_steps: int) -> TrainConfig:
        return TrainConfig(
            ppo=PPOConfig(n_envs=4, rollout_length=20, total_steps=total_steps),
            eval=EvalConfig(enabled=False),
            checkpoint_every_steps=checkpoint_every_steps,
        )

    def test_checkpoint_fn_called_at_step_zero(self):
        """A checkpoint fires at step 0 because last_checkpoint_step starts at -every_steps."""
        calls = []
        config = self._tiny_config(total_steps=4 * 20, checkpoint_every_steps=10_000_000)
        ppo.train_ppo(
            self.env, self.nets, config,
            checkpoint_fn=lambda s, t: calls.append(t),
        )
        self.assertIn(0, calls, "Expected a checkpoint at step 0.")

    def test_checkpoint_fn_called_correct_number_of_times(self):
        N_ENVS = 4
        ROLLOUT = 20
        STEPS_PER_ITER = N_ENVS * ROLLOUT  # 80
        N_ITERS = 3
        calls = []
        config = self._tiny_config(
            total_steps=STEPS_PER_ITER * N_ITERS,
            checkpoint_every_steps=STEPS_PER_ITER,
        )
        ppo.train_ppo(
            self.env, self.nets, config,
            checkpoint_fn=lambda s, t: calls.append(t),
        )
        # Step-0 checkpoint + one per iteration = N_ITERS + 1 calls total.
        self.assertEqual(len(calls), N_ITERS + 1)
        self.assertEqual(calls[0], 0)

    def test_no_checkpoint_fn_runs_normally(self):
        config = self._tiny_config(total_steps=4 * 20, checkpoint_every_steps=1000)
        result = ppo.train_ppo(self.env, self.nets, config, checkpoint_fn=None)
        self.assertEqual(result.total_steps, 4 * 20)

    def test_checkpoint_fn_receives_training_state(self):
        received = []
        config = self._tiny_config(
            total_steps=4 * 20,
            checkpoint_every_steps=10_000_000,  # only fires at step 0
        )
        ppo.train_ppo(
            self.env, self.nets, config,
            checkpoint_fn=lambda s, t: received.append((s, t)),
        )
        self.assertGreater(len(received), 0)
        state, step = received[0]
        self.assertIsInstance(state, TrainingState)
        self.assertEqual(step, 0)

    def test_checkpoint_not_called_after_step_zero_when_every_steps_huge(self):
        """With a huge every_steps, only the step-0 checkpoint fires."""
        calls = []
        config = self._tiny_config(
            total_steps=4 * 20,
            checkpoint_every_steps=10_000_000,
        )
        ppo.train_ppo(
            self.env, self.nets, config,
            checkpoint_fn=lambda s, t: calls.append(t),
        )
        self.assertEqual(calls, [0])


if __name__ == "__main__":
    absltest.main()