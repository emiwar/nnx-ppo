from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.recurrent import LSTM
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.factories import make_mlp
from nnx_ppo.networks.containers import Sequential, PPOActorCritic
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.algorithms import rollout
from nnx_ppo.algorithms.ppo import ppo_step, new_training_state
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.test_dummies.mock_env import MockEnv


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
        lstm = LSTM(
            in_features=in_features, hidden_features=hidden_features, rngs=nnx.Rngs(42)
        )
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
            in_features=16,
            hidden_features=32,
            rngs=nnx.Rngs(42),
            trainable_initial_state=True,
        )
        self.assertTrue(lstm.trainable_initial_state)
        # Check parameters exist
        self.assertTrue(hasattr(lstm, "initial_h"))
        self.assertTrue(hasattr(lstm, "initial_c"))
        self.assertIsInstance(lstm.initial_h, nnx.Param)
        self.assertIsInstance(lstm.initial_c, nnx.Param)
        # Check shapes
        self.assertEqual(lstm.initial_h[...].shape, (32,))
        self.assertEqual(lstm.initial_c[...].shape, (32,))

    def test_trainable_initial_state_broadcasts(self):
        """Trainable initial state should broadcast to batch size."""
        lstm = LSTM(
            in_features=16,
            hidden_features=32,
            rngs=nnx.Rngs(42),
            trainable_initial_state=True,
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
            in_features=16,
            hidden_features=32,
            rngs=nnx.Rngs(42),
            trainable_initial_state=True,
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
        seq = Sequential(
            [
                make_mlp([in_features, 24], rngs),
                LSTM(in_features=24, hidden_features=hidden_features, rngs=rngs),
                make_mlp([hidden_features, output_features], rngs),
            ]
        )
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
        lstm_opt = LSTM(
            in_features=16, hidden_features=32, rngs=nnx.Rngs(42), use_optimized=True
        )
        lstm_std = LSTM(
            in_features=16, hidden_features=32, rngs=nnx.Rngs(42), use_optimized=False
        )
        self.assertIsInstance(lstm_opt.cell, nnx.OptimizedLSTMCell)
        self.assertIsInstance(lstm_std.cell, nnx.LSTMCell)

    def test_no_regularization_loss(self):
        """LSTM should have zero regularization loss."""
        lstm = LSTM(in_features=16, hidden_features=32, rngs=nnx.Rngs(42))
        state = lstm.initialize_state(batch_size=4)
        x = jp.ones((4, 16))
        output = lstm(state, x)
        self.assertTrue(jp.allclose(output.regularization_loss, jp.zeros(4)))

    def test_rollout_with_resets(self):
        """LSTM should work correctly during rollout with environment resets."""
        rngs = nnx.Rngs(42)
        obs_size = 16
        hidden_size = 32
        action_size = 4
        batch_size = 8
        unroll_length = 20
        max_steps = 5  # Env resets every 5 steps

        env = MockEnv(obs_size, action_size, max_steps=max_steps)

        # Create network with LSTM in actor
        actor = Sequential(
            [
                Dense(obs_size, hidden_size, rngs, activation=nnx.relu),
                LSTM(in_features=hidden_size, hidden_features=hidden_size, rngs=rngs),
                Dense(hidden_size, action_size * 2, rngs, activation=None),
            ]
        )

        critic = Sequential(
            [
                Dense(obs_size, hidden_size, rngs, activation=nnx.relu),
                Dense(hidden_size, 1, rngs, activation=None),
            ]
        )

        sampler = NormalTanhSampler(rngs, entropy_weight=1e-3)
        networks = PPOActorCritic(actor=actor, critic=critic, action_sampler=sampler)

        # Initialize states
        key = jax.random.PRNGKey(0)
        env_keys = jax.random.split(key, batch_size)
        env_states = jax.vmap(env.reset)(env_keys)
        network_states = networks.initialize_state(batch_size)

        # Run rollout
        rollout_key = jax.random.PRNGKey(1)
        final_net_state, final_env_state, rollout_data = rollout.unroll_env(
            env, env_states, networks, network_states, unroll_length, rollout_key
        )

        # Check that multiple resets occurred
        total_dones = jp.sum(rollout_data.done)
        self.assertGreater(
            float(total_dones), 0, "Expected at least one reset to occur"
        )

        # Check no NaNs in network outputs
        self.assertFalse(jp.any(jp.isnan(rollout_data.network_output.actions)))
        self.assertFalse(jp.any(jp.isnan(rollout_data.network_output.value_estimates)))
        self.assertFalse(jp.any(jp.isnan(rollout_data.network_output.loglikelihoods)))

    def test_ppo_step_with_lstm(self):
        """LSTM should work correctly during ppo_step including backward pass."""
        rngs = nnx.Rngs(42)
        obs_size = 16
        hidden_size = 32
        action_size = 4
        n_envs = 8
        rollout_length = 20
        max_steps = 5

        env = MockEnv(obs_size, action_size, max_steps=max_steps)

        actor = Sequential(
            [
                Dense(obs_size, hidden_size, rngs, activation=nnx.relu),
                LSTM(in_features=hidden_size, hidden_features=hidden_size, rngs=rngs),
                Dense(hidden_size, action_size * 2, rngs, activation=None),
            ]
        )

        critic = Sequential(
            [
                Dense(obs_size, hidden_size, rngs, activation=nnx.relu),
                Dense(hidden_size, 1, rngs, activation=None),
            ]
        )

        sampler = NormalTanhSampler(rngs, entropy_weight=1e-3)
        networks = PPOActorCritic(actor=actor, critic=critic, action_sampler=sampler)

        training_state = new_training_state(
            env, networks, n_envs, seed=42, learning_rate=1e-4, gradient_clipping=1.0
        )

        new_state, metrics = ppo_step(
            env,
            training_state,
            n_envs=n_envs,
            rollout_length=rollout_length,
            gae_lambda=0.95,
            discounting_factor=0.99,
            clip_range=0.2,
            normalize_advantages=True,
            n_epochs=2,
            n_minibatches=2,
            logging_level=LoggingLevel.LOSSES,
        )

        # Check no NaNs in losses
        self.assertFalse(jp.any(jp.isnan(metrics["losses/actor/mean"])))
        self.assertFalse(jp.any(jp.isnan(metrics["losses/critic/mean"])))
        self.assertFalse(jp.any(jp.isnan(metrics["losses/regularization/mean"])))

        # Check no NaNs in network parameters after update
        params = nnx.state(new_state.networks, nnx.Param)
        for leaf in jax.tree.leaves(params):
            self.assertFalse(jp.any(jp.isnan(leaf)))


if __name__ == "__main__":
    absltest.main()
