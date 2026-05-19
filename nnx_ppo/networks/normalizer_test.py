from absl.testing import absltest
import jax
import jax.numpy as jp

from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.types import Context


class NormalizerTest(absltest.TestCase):

    def test_normalizer_initialization(self):
        normalizer = Normalizer(10)
        self.assertEqual(normalizer.mean.get_value().shape, (10,))
        self.assertEqual(normalizer.M2.get_value().shape, (10,))
        self.assertEqual(normalizer.counter.get_value(), 0.0)

    def test_normalizer_tuple_shape(self):
        normalizer = Normalizer((5, 3))
        self.assertEqual(normalizer.mean.get_value().shape, (5, 3))
        self.assertEqual(normalizer.M2.get_value().shape, (5, 3))

    def test_normalizer_initial_call(self):
        """Before any statistics are collected, normalizer uses default std."""
        normalizer = Normalizer(4)
        state = normalizer.initialize_state(1)
        x = jp.array([[1.0, 2.0, 3.0, 4.0]])
        output = normalizer(state, x)
        # With counter=0, std defaults to 10.0
        expected = (x - 0.0) / 10.0
        self.assertTrue(jp.allclose(output.output, expected))

    def test_normalizer_stats_update_context(self):
        SEED = 42
        OBS_SIZE = 8
        BATCH_SIZE = 16
        N_STEPS = 5

        normalizer = Normalizer(OBS_SIZE)
        key = jax.random.key(SEED)
        mean_key, var_key = jax.random.split(key)
        true_mean = jax.random.normal(mean_key, (OBS_SIZE,))
        data = true_mean + jax.random.normal(var_key, (N_STEPS, BATCH_SIZE, OBS_SIZE))

        state = normalizer.initialize_state(BATCH_SIZE)
        for i in range(N_STEPS):
            normalizer(state, data[i], context=Context.STATS_UPDATE)
            self.assertEqual(normalizer.counter.get_value(), (i + 1) * BATCH_SIZE)

        true_data_mean = jp.mean(data, axis=(0, 1))
        true_data_std = jp.std(data, axis=(0, 1))
        est_mean = normalizer.mean.get_value()
        est_std = jp.sqrt(normalizer.M2.get_value() / normalizer.counter.get_value())

        self.assertLess(jp.max(jp.abs(est_mean - true_data_mean)), 1e-5)
        self.assertLess(jp.max(jp.abs(est_std - true_data_std)), 1e-5)

    def test_normalizer_normalization(self):
        """Test that normalization produces zero mean, unit variance output."""
        SEED = 123
        OBS_SIZE = 6
        BATCH_SIZE = 64
        N_STEPS = 20

        normalizer = Normalizer(OBS_SIZE)
        key = jax.random.key(SEED)
        data_key, test_key = jax.random.split(key)

        true_mean = jp.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
        true_std = jp.array([0.5, 1.0, 2.0, 0.1, 3.0, 0.8])
        data = true_mean + true_std * jax.random.normal(
            data_key, (N_STEPS, BATCH_SIZE, OBS_SIZE)
        )

        state = normalizer.initialize_state(BATCH_SIZE)
        for i in range(N_STEPS):
            normalizer(state, data[i], context=Context.STATS_UPDATE)

        # Generate test data from same distribution
        test_data = true_mean + true_std * jax.random.normal(test_key, (100, OBS_SIZE))
        state = normalizer.initialize_state(100)
        output = normalizer(state, test_data)

        # Normalized output should have approximately zero mean and unit variance
        normalized_mean = jp.mean(output.output, axis=0)
        normalized_std = jp.std(output.output, axis=0)

        self.assertTrue(jp.all(jp.abs(normalized_mean) < 0.3))
        self.assertTrue(jp.all(jp.abs(normalized_std - 1.0) < 0.3))


class NormalizerPytreeTest(absltest.TestCase):
    """Tests for Normalizer with dict-structured (pytree) observations.

    This mirrors the rodent_enc_dec.py usage where non_flattened_observation_size
    returns a dict like {"proprioception": ..., "imitation_target": ...}.
    """

    OBS_SIZE = {
        "proprioception": 8,
        "goal": 4,
    }

    def _make_dict_obs(self, batch: int, key) -> dict:
        k1, k2 = jax.random.split(key)
        return {
            "proprioception": jax.random.normal(k1, (batch, 8)),
            "goal": jax.random.normal(k2, (batch, 4)),
        }

    def test_pytree_initialization(self):
        normalizer = Normalizer(self.OBS_SIZE)
        mean = normalizer.mean.get_value()
        self.assertIsInstance(mean, dict)
        self.assertEqual(set(mean.keys()), {"proprioception", "goal"})
        self.assertEqual(mean["proprioception"].shape, (8,))
        self.assertEqual(mean["goal"].shape, (4,))

    def test_pytree_forward_pass_before_update(self):
        """Before any update, default std=10 is used for each leaf."""
        normalizer = Normalizer(self.OBS_SIZE)
        state = normalizer.initialize_state(1)
        obs = {
            "proprioception": jp.full((1, 8), 2.0),
            "goal": jp.full((1, 4), 5.0),
        }
        output = normalizer(state, obs)
        self.assertIsInstance(output.output, dict)
        # With counter=0, std defaults to 10.0 and mean=0
        self.assertTrue(jp.allclose(output.output["proprioception"], jp.full((1, 8), 0.2)))
        self.assertTrue(jp.allclose(output.output["goal"], jp.full((1, 4), 0.5)))

    def test_pytree_stats_update(self):
        """STATS_UPDATE context should accumulate stats over dict observations."""
        SEED = 7
        BATCH_SIZE = 32
        N_STEPS = 10
        normalizer = Normalizer(self.OBS_SIZE)

        state = normalizer.initialize_state(BATCH_SIZE)
        key = jax.random.key(SEED)
        for _ in range(N_STEPS):
            key, subkey = jax.random.split(key)
            obs = self._make_dict_obs(BATCH_SIZE, subkey)
            normalizer(state, obs, context=Context.STATS_UPDATE)

        self.assertEqual(normalizer.counter.get_value(), N_STEPS * BATCH_SIZE)
        for leaf in jax.tree.leaves(normalizer.mean.get_value()):
            self.assertLess(float(jp.max(jp.abs(leaf))), 0.5)

    def test_pytree_normalization(self):
        """After update, normalizer should produce approximately zero-mean unit-var output."""
        SEED = 99
        BATCH_SIZE = 64
        N_STEPS = 20
        normalizer = Normalizer(self.OBS_SIZE)

        true_mean = {"proprioception": jp.full((8,), 3.0), "goal": jp.full((4,), -2.0)}
        state = normalizer.initialize_state(BATCH_SIZE)
        key = jax.random.key(SEED)
        for _ in range(N_STEPS):
            key, subkey = jax.random.split(key)
            raw = self._make_dict_obs(BATCH_SIZE, subkey)
            obs = jax.tree.map(lambda x, m: x + m, raw, true_mean)
            normalizer(state, obs, context=Context.STATS_UPDATE)

        key, subkey = jax.random.split(key)
        test_raw = self._make_dict_obs(200, subkey)
        test_obs = jax.tree.map(lambda x, m: x + m, test_raw, true_mean)
        eval_state = normalizer.initialize_state(200)
        output = normalizer(eval_state, test_obs)
        for key_name, leaf in output.output.items():
            self.assertLess(float(jp.max(jp.abs(jp.mean(leaf, axis=0)))), 0.5,
                            msg=f"key={key_name}: normalized mean too far from 0")


class NormalizerStatsUpdateContextTest(absltest.TestCase):
    """STATS_UPDATE context updates live stats; other contexts must not."""

    def test_non_stats_update_context_does_not_change_stats(self):
        """Calling __call__ with ROLLOUT / LOSS_REPLAY / INFERENCE must not
        mutate the live mean/M2/counter."""
        normalizer = Normalizer(3)
        # Seed with some data first via STATS_UPDATE.
        state = normalizer.initialize_state(4)
        normalizer(state, jp.ones((4, 3)) * 5.0, context=Context.STATS_UPDATE)
        snapshot_mean = normalizer.mean.get_value()
        snapshot_M2 = normalizer.M2.get_value()
        snapshot_count = normalizer.counter.get_value()

        for ctx in (Context.ROLLOUT, Context.LOSS_REPLAY, Context.INFERENCE):
            normalizer(state, jp.ones((4, 3)) * 999.0, context=ctx)

        self.assertTrue(jp.allclose(normalizer.mean.get_value(), snapshot_mean))
        self.assertTrue(jp.allclose(normalizer.M2.get_value(), snapshot_M2))
        self.assertEqual(
            float(normalizer.counter.get_value()), float(snapshot_count)
        )


if __name__ == "__main__":
    absltest.main()
