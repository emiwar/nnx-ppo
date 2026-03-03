from absl.testing import absltest
import jax
import jax.numpy as jp

from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.algorithms.types import Transition
from nnx_ppo.networks.types import PPONetworkOutput


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
            value_estimates=jp.zeros((BATCH_SIZE,)),
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
            self.assertEqual(normalizer.counter.get_value(), (i + 1) * BATCH_SIZE)

        # After all updates, check statistics accuracy
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
            value_estimates=jp.zeros((BATCH_SIZE,)),
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


class NormalizerPytreeTest(absltest.TestCase):
    """Tests for Normalizer with dict-structured (pytree) observations.

    This mirrors the rodent_enc_dec.py usage where non_flattened_observation_size
    returns a dict like {"proprioception": ..., "imitation_target": ...}.
    """

    OBS_SIZE = {
        "proprioception": 8,
        "goal": 4,
    }

    def _make_dict_obs(self, n_time: int, batch: int, key) -> dict:
        k1, k2 = jax.random.split(key)
        return {
            "proprioception": jax.random.normal(k1, (n_time, batch, 8)),
            "goal": jax.random.normal(k2, (n_time, batch, 4)),
        }

    def _make_transition(self, obs, batch_size) -> Transition:
        dummy_output = PPONetworkOutput(
            actions=jp.zeros((batch_size, 1)),
            raw_actions=jp.zeros((batch_size, 1)),
            loglikelihoods=jp.zeros((batch_size,)),
            regularization_loss=jp.array(0.0),
            value_estimates=jp.zeros((batch_size,)),
            metrics={},
        )
        n_time = jax.tree.leaves(obs)[0].shape[0]
        return Transition(
            obs=obs,
            network_output=dummy_output,
            rewards=jp.zeros((n_time, batch_size)),
            done=jp.zeros((n_time, batch_size), jp.bool),
            truncated=jp.zeros((n_time, batch_size), jp.bool),
            next_obs=jax.tree.map(jp.zeros_like, obs),
            metrics={},
        )

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

    def test_pytree_update_statistics(self):
        """update_statistics should work with dict-structured observations."""
        SEED = 7
        BATCH_SIZE = 32
        N_STEPS = 10
        normalizer = Normalizer(self.OBS_SIZE)

        key = jax.random.key(SEED)
        for i in range(N_STEPS):
            key, subkey = jax.random.split(key)
            obs = self._make_dict_obs(1, BATCH_SIZE, subkey)
            transition = self._make_transition(obs, BATCH_SIZE)
            normalizer.update_statistics(transition, (i + 1) * BATCH_SIZE)

        self.assertEqual(normalizer.counter.get_value(), N_STEPS * BATCH_SIZE)
        # mean should be approximately 0 for standard normal data
        for leaf in jax.tree.leaves(normalizer.mean.get_value()):
            self.assertLess(float(jp.max(jp.abs(leaf))), 0.5)

    def test_pytree_normalization(self):
        """After update, normalizer should produce approximately zero-mean unit-var output."""
        SEED = 99
        BATCH_SIZE = 64
        N_STEPS = 20
        normalizer = Normalizer(self.OBS_SIZE)

        # Shift obs so normalization has real work to do
        true_mean = {"proprioception": jp.full((8,), 3.0), "goal": jp.full((4,), -2.0)}
        key = jax.random.key(SEED)
        for i in range(N_STEPS):
            key, subkey = jax.random.split(key)
            raw = self._make_dict_obs(1, BATCH_SIZE, subkey)
            obs = jax.tree.map(lambda x, m: x + m, raw, true_mean)
            transition = self._make_transition(obs, BATCH_SIZE)
            normalizer.update_statistics(transition, (i + 1) * BATCH_SIZE)

        # Test that a fresh batch gets normalized to ~zero mean
        key, subkey = jax.random.split(key)
        test_raw = self._make_dict_obs(1, 200, subkey)
        test_obs = jax.tree.map(lambda x, m: x + m, test_raw, true_mean)
        # Squeeze time dim for the forward pass
        test_obs_squeezed = jax.tree.map(lambda x: x[0], test_obs)
        state = normalizer.initialize_state(200)
        output = normalizer(state, test_obs_squeezed)
        for key_name, leaf in output.output.items():
            self.assertLess(float(jp.max(jp.abs(jp.mean(leaf, axis=0)))), 0.5,
                            msg=f"key={key_name}: normalized mean too far from 0")


if __name__ == "__main__":
    absltest.main()
