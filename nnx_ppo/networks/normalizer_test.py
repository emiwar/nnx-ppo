from absl.testing import absltest
import jax
import jax.numpy as jp

from nnx_ppo.networks.normalizer import Normalizer



def _collect_rollout_extras(normalizer, batch_data):
    """Run ROLLOUT forward over N steps, return stacked rollout_extras."""
    state = normalizer.initialize_state(batch_data.shape[1])
    per_step = []
    for t in range(batch_data.shape[0]):
        out = normalizer(state, batch_data[t])
        per_step.append(out.rollout_extras)
        state = out.next_state
    return jax.tree.map(lambda *xs: jp.stack(xs, axis=0), *per_step)


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
        expected = (x - 0.0) / 10.0
        self.assertTrue(jp.allclose(output.output, expected))

    def test_update_statistics_matches_true_moments(self):
        SEED = 42
        OBS_SIZE = 8
        BATCH_SIZE = 16
        N_STEPS = 5

        normalizer = Normalizer(OBS_SIZE)
        key = jax.random.key(SEED)
        mean_key, var_key = jax.random.split(key)
        true_mean = jax.random.normal(mean_key, (OBS_SIZE,))
        data = true_mean + jax.random.normal(var_key, (N_STEPS, BATCH_SIZE, OBS_SIZE))

        rollout_extras = _collect_rollout_extras(normalizer, data)
        normalizer.update_statistics(rollout_extras)

        self.assertEqual(
            float(normalizer.counter.get_value()), N_STEPS * BATCH_SIZE
        )
        true_data_mean = jp.mean(data, axis=(0, 1))
        true_data_std = jp.std(data, axis=(0, 1))
        est_mean = normalizer.mean.get_value()
        est_std = jp.sqrt(normalizer.M2.get_value() / normalizer.counter.get_value())
        self.assertLess(float(jp.max(jp.abs(est_mean - true_data_mean))), 1e-5)
        self.assertLess(float(jp.max(jp.abs(est_std - true_data_std))), 1e-5)

    def test_normalizer_normalization_after_update(self):
        """Normalization produces approximately zero mean / unit variance output."""
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

        rollout_extras = _collect_rollout_extras(normalizer, data)
        normalizer.update_statistics(rollout_extras)

        test_data = true_mean + true_std * jax.random.normal(test_key, (100, OBS_SIZE))
        state = normalizer.initialize_state(100)
        output = normalizer(state, test_data)

        normalized_mean = jp.mean(output.output, axis=0)
        normalized_std = jp.std(output.output, axis=0)
        self.assertTrue(jp.all(jp.abs(normalized_mean) < 0.3))
        self.assertTrue(jp.all(jp.abs(normalized_std - 1.0) < 0.3))


class NormalizerPytreeTest(absltest.TestCase):

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
        normalizer = Normalizer(self.OBS_SIZE)
        state = normalizer.initialize_state(1)
        obs = {
            "proprioception": jp.full((1, 8), 2.0),
            "goal": jp.full((1, 4), 5.0),
        }
        output = normalizer(state, obs)
        self.assertIsInstance(output.output, dict)
        self.assertTrue(jp.allclose(output.output["proprioception"], jp.full((1, 8), 0.2)))
        self.assertTrue(jp.allclose(output.output["goal"], jp.full((1, 4), 0.5)))

    def test_pytree_update_statistics(self):
        SEED = 7
        BATCH_SIZE = 32
        N_STEPS = 10
        normalizer = Normalizer(self.OBS_SIZE)

        state = normalizer.initialize_state(BATCH_SIZE)
        key = jax.random.key(SEED)
        per_step = []
        for _ in range(N_STEPS):
            key, subkey = jax.random.split(key)
            obs = self._make_dict_obs(BATCH_SIZE, subkey)
            out = normalizer(state, obs)
            per_step.append(out.rollout_extras)
            state = out.next_state
        rollout_extras = jax.tree.map(lambda *xs: jp.stack(xs, axis=0), *per_step)
        normalizer.update_statistics(rollout_extras)

        self.assertEqual(
            float(normalizer.counter.get_value()), N_STEPS * BATCH_SIZE
        )
        for leaf in jax.tree.leaves(normalizer.mean.get_value()):
            self.assertLess(float(jp.max(jp.abs(leaf))), 0.5)


class NormalizerCallNeverWritesTest(absltest.TestCase):
    """Forward `__call__` must not mutate live mean/M2/counter."""

    def test_call_does_not_change_stats(self):
        normalizer = Normalizer(3)
        # Seed with some data via update_statistics.
        # rollout_extras shape: [T=1, B=4, *feat=(3,)]
        normalizer.update_statistics(jp.ones((1, 4, 3)) * 5.0)
        snapshot_mean = normalizer.mean.get_value()
        snapshot_M2 = normalizer.M2.get_value()
        snapshot_count = normalizer.counter.get_value()

        state = normalizer.initialize_state(4)
        # Try both with and without rollout_extras passed in.
        normalizer(state, jp.ones((4, 3)) * 999.0)
        normalizer(state, jp.ones((4, 3)) * 999.0, jp.ones((4, 3)) * 1.0)

        self.assertTrue(jp.allclose(normalizer.mean.get_value(), snapshot_mean))
        self.assertTrue(jp.allclose(normalizer.M2.get_value(), snapshot_M2))
        self.assertEqual(
            float(normalizer.counter.get_value()), float(snapshot_count)
        )


if __name__ == "__main__":
    absltest.main()
