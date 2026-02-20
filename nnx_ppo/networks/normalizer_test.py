from absl.testing import absltest
import jax
import jax.numpy as jp

from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.algorithms.types import Transition
from nnx_ppo.networks.types import PPONetworkOutput


class NormalizerTest(absltest.TestCase):

    def test_normalizer_initialization(self):
        normalizer = Normalizer(10)
        self.assertEqual(normalizer.mean[...].shape, (10,))
        self.assertEqual(normalizer.M2[...].shape, (10,))
        self.assertEqual(normalizer.counter[...], 0.0)

    def test_normalizer_tuple_shape(self):
        normalizer = Normalizer((5, 3))
        self.assertEqual(normalizer.mean[...].shape, (5, 3))
        self.assertEqual(normalizer.M2[...].shape, (5, 3))

    def test_normalizer_initial_call(self):
        """Before any statistics are collected, normalizer uses default std."""
        normalizer = Normalizer(4)
        state = normalizer.initialize_state(1)
        x = jp.array([[1.0, 2.0, 3.0, 4.0]])
        output = normalizer(state, x)
        # With counter=0, std defaults to 10.0
        expected = (x - 0.0) / 10.0
        self.assertTrue(jp.allclose(output.output, expected))

    def test_normalizer_update_statistics(self):
        SEED = 42
        OBS_SIZE = 8
        BATCH_SIZE = 16
        N_STEPS = 5

        normalizer = Normalizer(OBS_SIZE)
        key = jax.random.key(SEED)
        mean_key, var_key = jax.random.split(key)
        true_mean = jax.random.normal(mean_key, (OBS_SIZE,))
        data = true_mean + jax.random.normal(var_key, (N_STEPS, BATCH_SIZE, OBS_SIZE))

        # Create dummy PPONetworkOutput for Transition
        dummy_network_output = PPONetworkOutput(
            actions=jp.zeros((BATCH_SIZE, 1)),
            raw_actions=jp.zeros((BATCH_SIZE, 1)),
            loglikelihoods=jp.zeros((BATCH_SIZE,)),
            regularization_loss=jp.array(0.0),
            value_estimates=jp.zeros((BATCH_SIZE, 1)),
            metrics={},
        )

        for i in range(N_STEPS):
            obs_batch = data[i : i + 1]  # Shape: (1, BATCH_SIZE, OBS_SIZE)
            dummy_transition = Transition(
                obs=obs_batch,
                network_output=dummy_network_output,
                rewards=jp.zeros((1, BATCH_SIZE)),
                done=jp.zeros((1, BATCH_SIZE), jp.bool),
                truncated=jp.zeros((1, BATCH_SIZE), jp.bool),
                next_obs=jp.zeros_like(obs_batch),
                metrics={},
            )
            normalizer.update_statistics(dummy_transition, (i + 1) * BATCH_SIZE)
            self.assertEqual(normalizer.counter[...], (i + 1) * BATCH_SIZE)

        # After all updates, check statistics accuracy
        true_data_mean = jp.mean(data, axis=(0, 1))
        true_data_std = jp.std(data, axis=(0, 1))
        est_mean = normalizer.mean[...]
        est_std = jp.sqrt(normalizer.M2[...] / normalizer.counter[...])

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

        # Generate training data with non-trivial mean and variance
        true_mean = jp.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
        true_std = jp.array([0.5, 1.0, 2.0, 0.1, 3.0, 0.8])
        data = true_mean + true_std * jax.random.normal(
            data_key, (N_STEPS, BATCH_SIZE, OBS_SIZE)
        )

        dummy_network_output = PPONetworkOutput(
            actions=jp.zeros((BATCH_SIZE, 1)),
            raw_actions=jp.zeros((BATCH_SIZE, 1)),
            loglikelihoods=jp.zeros((BATCH_SIZE,)),
            regularization_loss=jp.array(0.0),
            value_estimates=jp.zeros((BATCH_SIZE, 1)),
            metrics={},
        )

        # Update statistics
        for i in range(N_STEPS):
            obs_batch = data[i : i + 1]
            dummy_transition = Transition(
                obs=obs_batch,
                network_output=dummy_network_output,
                rewards=jp.zeros((1, BATCH_SIZE)),
                done=jp.zeros((1, BATCH_SIZE), jp.bool),
                truncated=jp.zeros((1, BATCH_SIZE), jp.bool),
                next_obs=jp.zeros_like(obs_batch),
                metrics={},
            )
            normalizer.update_statistics(dummy_transition, (i + 1) * BATCH_SIZE)

        # Generate test data from same distribution
        test_data = true_mean + true_std * jax.random.normal(test_key, (100, OBS_SIZE))
        state = normalizer.initialize_state(100)
        output = normalizer(state, test_data)

        # Normalized output should have approximately zero mean and unit variance
        normalized_mean = jp.mean(output.output, axis=0)
        normalized_std = jp.std(output.output, axis=0)

        self.assertTrue(jp.all(jp.abs(normalized_mean) < 0.3))
        self.assertTrue(jp.all(jp.abs(normalized_std - 1.0) < 0.3))


if __name__ == "__main__":
    absltest.main()
