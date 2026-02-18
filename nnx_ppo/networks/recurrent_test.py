from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.recurrent import LSTM
from nnx_ppo.networks.feedforward import MLP
from nnx_ppo.networks.containers import Sequential


class LSTMTest(absltest.TestCase):

    def test_initialization(self):
        lstm = LSTM(in_features=16, hidden_features=32, rngs=nnx.Rngs(42))
        self.assertEqual(lstm.in_features, 16)
        self.assertEqual(lstm.hidden_features, 32)
        self.assertFalse(lstm.trainable_initial_state)

    def test_output_shape(self):
        in_features = 16
        hidden_features = 32
        batch_size = 4
        lstm = LSTM(in_features=in_features, hidden_features=hidden_features, rngs=nnx.Rngs(42))
        state = lstm.initialize_state(batch_size)
        x = jp.ones((batch_size, in_features))
        output = lstm(state, x)
        self.assertEqual(output.output.shape, (batch_size, hidden_features))
        self.assertEqual(output.regularization_loss.shape, (batch_size,))

    def test_state_is_hc_tuple(self):
        """State should be a tuple of (h, c) arrays."""
        lstm = LSTM(in_features=16, hidden_features=32, rngs=nnx.Rngs(42))
        state = lstm.initialize_state(batch_size=4)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 2)
        h, c = state
        self.assertEqual(h.shape, (4, 32))
        self.assertEqual(c.shape, (4, 32))

    def test_initial_state_is_zeros(self):
        """Default initial state should be zeros."""
        lstm = LSTM(in_features=16, hidden_features=32, rngs=nnx.Rngs(42))
        state = lstm.initialize_state(batch_size=4)
        h, c = state
        self.assertTrue(jp.allclose(h, jp.zeros_like(h)))
        self.assertTrue(jp.allclose(c, jp.zeros_like(c)))

    def test_state_updates_through_timesteps(self):
        """State should change after processing input."""
        lstm = LSTM(in_features=16, hidden_features=32, rngs=nnx.Rngs(42))
        state = lstm.initialize_state(batch_size=4)
        x = jp.ones((4, 16))
        output = lstm(state, x)
        h_new, c_new = output.next_state
        h_old, c_old = state
        # State should have changed
        self.assertFalse(jp.allclose(h_new, h_old))
        self.assertFalse(jp.allclose(c_new, c_old))

    def test_reset_state_returns_initial(self):
        """reset_state should return the initial state."""
        lstm = LSTM(in_features=16, hidden_features=32, rngs=nnx.Rngs(42))
        state = lstm.initialize_state(batch_size=4)
        x = jp.ones((4, 16))
        # Run a few steps
        for _ in range(3):
            output = lstm(state, x)
            state = output.next_state
        # Reset
        reset_state = lstm.reset_state(state)
        h, c = reset_state
        # Should be zeros again
        self.assertTrue(jp.allclose(h, jp.zeros_like(h)))
        self.assertTrue(jp.allclose(c, jp.zeros_like(c)))

    def test_trainable_initial_state(self):
        """Trainable initial state should be learnable parameters."""
        lstm = LSTM(
            in_features=16, hidden_features=32, rngs=nnx.Rngs(42),
            trainable_initial_state=True
        )
        self.assertTrue(lstm.trainable_initial_state)
        # Check parameters exist
        self.assertTrue(hasattr(lstm, 'initial_h'))
        self.assertTrue(hasattr(lstm, 'initial_c'))
        self.assertIsInstance(lstm.initial_h, nnx.Param)
        self.assertIsInstance(lstm.initial_c, nnx.Param)
        # Check shapes
        self.assertEqual(lstm.initial_h[...].shape, (32,))
        self.assertEqual(lstm.initial_c[...].shape, (32,))

    def test_trainable_initial_state_broadcasts(self):
        """Trainable initial state should broadcast to batch size."""
        lstm = LSTM(
            in_features=16, hidden_features=32, rngs=nnx.Rngs(42),
            trainable_initial_state=True
        )
        # Modify the initial state to non-zero values
        lstm.initial_h[...] = jp.ones(32) * 0.5
        lstm.initial_c[...] = jp.ones(32) * 0.3
        state = lstm.initialize_state(batch_size=4)
        h, c = state
        self.assertEqual(h.shape, (4, 32))
        self.assertEqual(c.shape, (4, 32))
        # All batch elements should have same initial values
        self.assertTrue(jp.allclose(h, jp.ones((4, 32)) * 0.5))
        self.assertTrue(jp.allclose(c, jp.ones((4, 32)) * 0.3))

    def test_trainable_initial_state_reset(self):
        """reset_state with trainable initial state should return learned values."""
        lstm = LSTM(
            in_features=16, hidden_features=32, rngs=nnx.Rngs(42),
            trainable_initial_state=True
        )
        lstm.initial_h[...] = jp.ones(32) * 0.5
        lstm.initial_c[...] = jp.ones(32) * 0.3
        # Run some steps
        state = lstm.initialize_state(batch_size=4)
        x = jp.ones((4, 16))
        for _ in range(3):
            output = lstm(state, x)
            state = output.next_state
        # Reset should return to learned initial state
        reset_state = lstm.reset_state(state)
        h, c = reset_state
        self.assertTrue(jp.allclose(h, jp.ones((4, 32)) * 0.5))
        self.assertTrue(jp.allclose(c, jp.ones((4, 32)) * 0.3))

    def test_deterministic_given_state(self):
        """Same input and state should produce same output."""
        lstm = LSTM(in_features=16, hidden_features=32, rngs=nnx.Rngs(42))
        state = lstm.initialize_state(batch_size=4)
        x = jp.ones((4, 16))
        output1 = lstm(state, x)
        output2 = lstm(state, x)
        self.assertTrue(jp.allclose(output1.output, output2.output))
        self.assertTrue(jp.allclose(output1.next_state[0], output2.next_state[0]))
        self.assertTrue(jp.allclose(output1.next_state[1], output2.next_state[1]))

    def test_sequential_integration(self):
        """LSTM should work within Sequential container."""
        rngs = nnx.Rngs(42)
        in_features = 16
        hidden_features = 32
        output_features = 8
        batch_size = 4
        seq = Sequential([
            MLP([in_features, 24], rngs),
            LSTM(in_features=24, hidden_features=hidden_features, rngs=rngs),
            MLP([hidden_features, output_features], rngs),
        ])
        state = seq.initialize_state(batch_size)
        x = jp.ones((batch_size, in_features))
        output = seq(state, x)
        self.assertEqual(output.output.shape, (batch_size, output_features))

    def test_minibatch_slicing(self):
        """Slicing state for minibatches should preserve determinism per-env."""
        lstm = LSTM(in_features=16, hidden_features=32, rngs=nnx.Rngs(42))
        batch_size = 8
        x = jp.ones((batch_size, 16))

        # Run one step to get non-trivial state
        state = lstm.initialize_state(batch_size)
        output = lstm(state, x)
        state = output.next_state

        # Full batch second step
        x2 = jp.ones((batch_size, 16)) * 2
        output_full = lstm(state, x2)

        # Minibatch: first half
        state_mb1 = (state[0][:4], state[1][:4])
        x_mb1 = x2[:4]
        output_mb1 = lstm(state_mb1, x_mb1)

        # Minibatch: second half
        state_mb2 = (state[0][4:], state[1][4:])
        x_mb2 = x2[4:]
        output_mb2 = lstm(state_mb2, x_mb2)

        # Outputs should match the corresponding slices of full batch
        self.assertTrue(jp.allclose(output_full.output[:4], output_mb1.output))
        self.assertTrue(jp.allclose(output_full.output[4:], output_mb2.output))

    def test_use_optimized_flag(self):
        """use_optimized=False should use standard LSTMCell."""
        lstm_opt = LSTM(in_features=16, hidden_features=32, rngs=nnx.Rngs(42), use_optimized=True)
        lstm_std = LSTM(in_features=16, hidden_features=32, rngs=nnx.Rngs(42), use_optimized=False)
        self.assertIsInstance(lstm_opt.cell, nnx.OptimizedLSTMCell)
        self.assertIsInstance(lstm_std.cell, nnx.LSTMCell)

    def test_no_regularization_loss(self):
        """LSTM should have zero regularization loss."""
        lstm = LSTM(in_features=16, hidden_features=32, rngs=nnx.Rngs(42))
        state = lstm.initialize_state(batch_size=4)
        x = jp.ones((4, 16))
        output = lstm(state, x)
        self.assertTrue(jp.allclose(output.regularization_loss, jp.zeros(4)))


if __name__ == '__main__':
    absltest.main()
