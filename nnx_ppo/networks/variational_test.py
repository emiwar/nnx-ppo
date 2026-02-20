from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.variational import VariationalBottleneck
from nnx_ppo.networks.factories import make_mlp
from nnx_ppo.networks.containers import Sequential


class VariationalBottleneckTest(absltest.TestCase):

    def test_initialization(self):
        vb = VariationalBottleneck(latent_size=8, rng=nnx.Rngs(42), kl_weight=0.01)
        self.assertEqual(vb.latent_size, 8)
        self.assertEqual(vb.kl_weight, 0.01)

    def test_output_shape(self):
        latent_size = 8
        batch_size = 4
        vb = VariationalBottleneck(latent_size=latent_size, rng=nnx.Rngs(42))
        state = vb.initialize_state(batch_size)
        x = jp.ones((batch_size, latent_size * 2))
        output = vb(state, x)
        self.assertEqual(output.output.shape, (batch_size, latent_size))
        self.assertEqual(output.metrics["kl_divergence"].shape, (batch_size,))
        self.assertEqual(output.metrics["mu"].shape, (batch_size, latent_size))
        self.assertEqual(output.metrics["sigma"].shape, (batch_size, latent_size))

    def test_state_is_rng_keys(self):
        """State should be an array of RNG keys."""
        vb = VariationalBottleneck(latent_size=4, rng=nnx.Rngs(42))
        state = vb.initialize_state(batch_size=2)
        # State is array of keys with shape [batch_size, 2] (JAX key shape)
        self.assertEqual(state.shape[0], 2)
        x = jp.ones((2, 8))
        output = vb(state, x)
        # Next state should have same shape
        self.assertEqual(output.next_state.shape, state.shape)
        # But different values (keys advanced)
        self.assertFalse(jp.allclose(output.next_state, state))

    def test_kl_divergence_standard_normal(self):
        """KL divergence should be ~0 when input encodes standard normal."""
        latent_size = 16
        batch_size = 32
        vb = VariationalBottleneck(
            latent_size=latent_size, rng=nnx.Rngs(42), kl_weight=1.0
        )
        state = vb.initialize_state(batch_size)
        # Mean=0, softplus^-1(1 - min_std) ≈ softplus^-1(1) ≈ 0.54
        # For softplus(x) + min_std = 1, we need softplus(x) ≈ 1, so x ≈ 0.54
        x = jp.concatenate(
            [
                jp.zeros((batch_size, latent_size)),  # mean = 0
                jp.full((batch_size, latent_size), 0.54),  # softplus(0.54) ≈ 1
            ],
            axis=-1,
        )
        output = vb(state, x)
        # KL should be close to 0 for standard normal
        self.assertLess(jp.mean(output.metrics["kl_divergence"]), 0.1)

    def test_kl_divergence_increases_with_mean(self):
        """KL divergence should increase as mean deviates from 0."""
        latent_size = 8
        vb = VariationalBottleneck(
            latent_size=latent_size, rng=nnx.Rngs(42), kl_weight=1.0
        )
        state = vb.initialize_state(1)
        # Input with mean=0
        x_zero = jp.concatenate(
            [
                jp.zeros((1, latent_size)),
                jp.zeros((1, latent_size)),
            ],
            axis=-1,
        )
        output_zero = vb(state, x_zero)
        # Input with mean=5
        x_large = jp.concatenate(
            [
                jp.full((1, latent_size), 5.0),
                jp.zeros((1, latent_size)),
            ],
            axis=-1,
        )
        output_large = vb(state, x_large)
        self.assertGreater(
            float(output_large.metrics["kl_divergence"][0]),
            float(output_zero.metrics["kl_divergence"][0]),
        )

    def test_kl_weight_scales_loss(self):
        """Regularization loss should scale with kl_weight."""
        vb_low = VariationalBottleneck(latent_size=8, rng=nnx.Rngs(42), kl_weight=0.1)
        vb_high = VariationalBottleneck(latent_size=8, rng=nnx.Rngs(42), kl_weight=1.0)
        x = jp.ones((4, 16))
        state = vb_low.initialize_state(4)
        output_low = vb_low(state, x)
        output_high = vb_high(state, x)
        # regularization_loss is per-batch, so sum to compare
        ratio = jp.sum(output_high.regularization_loss) / jp.sum(
            output_low.regularization_loss
        )
        self.assertAlmostEqual(float(ratio), 10.0, places=5)

    def test_sampling_stochasticity(self):
        """Different RNG states should produce different samples."""
        latent_size = 8
        batch_size = 4
        x = jp.ones((batch_size, latent_size * 2))
        vb1 = VariationalBottleneck(latent_size=latent_size, rng=nnx.Rngs(42))
        vb2 = VariationalBottleneck(latent_size=latent_size, rng=nnx.Rngs(123))
        output1 = vb1(vb1.initialize_state(batch_size), x)
        output2 = vb2(vb2.initialize_state(batch_size), x)
        self.assertFalse(jp.allclose(output1.output, output2.output))

    def test_sequential_integration(self):
        """VariationalBottleneck should work within Sequential container."""
        rngs = nnx.Rngs(42)
        latent_size = 8
        obs_size = 16
        hidden_size = 32
        batch_size = 4
        seq = Sequential(
            [
                make_mlp(
                    [obs_size, hidden_size, latent_size * 2],
                    rngs,
                    activation_last_layer=False,
                ),
                VariationalBottleneck(latent_size, rngs, kl_weight=0.01),
                make_mlp([latent_size, hidden_size], rngs),
            ]
        )
        state = seq.initialize_state(batch_size)
        x = jp.ones((batch_size, obs_size))
        output = seq(state, x)
        self.assertEqual(output.output.shape, (batch_size, hidden_size))
        # Regularization loss should include KL term
        self.assertGreater(float(jp.sum(output.regularization_loss)), 0)

    def test_reset_state_preserves_keys(self):
        """reset_state should preserve the RNG keys."""
        vb = VariationalBottleneck(latent_size=4, rng=nnx.Rngs(42))
        state = vb.initialize_state(batch_size=2)
        reset_state = vb.reset_state(state)
        self.assertTrue(jp.allclose(state, reset_state))

    def test_minibatch_slicing(self):
        """Slicing state for minibatches should preserve determinism per-env."""
        latent_size = 8
        batch_size = 8
        vb = VariationalBottleneck(latent_size=latent_size, rng=nnx.Rngs(42))
        x = jp.ones((batch_size, latent_size * 2))

        # Full batch
        state = vb.initialize_state(batch_size)
        output_full = vb(state, x)

        # Minibatch: first half
        state_mb1 = state[:4]
        x_mb1 = x[:4]
        output_mb1 = vb(state_mb1, x_mb1)

        # Minibatch: second half
        state_mb2 = state[4:]
        x_mb2 = x[4:]
        output_mb2 = vb(state_mb2, x_mb2)

        # Samples should match the corresponding slices of full batch
        self.assertTrue(jp.allclose(output_full.output[:4], output_mb1.output))
        self.assertTrue(jp.allclose(output_full.output[4:], output_mb2.output))


if __name__ == "__main__":
    absltest.main()
