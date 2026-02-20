"""Tests for AR1VariationalBottleneck with environment rollouts and resets."""

from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.variational import AR1VariationalBottleneck
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.containers import PPOActorCritic, Sequential
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.algorithms import rollout
from nnx_ppo.algorithms.ppo import ppo_step, new_training_state
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.test_dummies.mock_env import MockEnv


class AR1RolloutTest(absltest.TestCase):

    def test_no_nan_leakage_during_rollout(self):
        """AR1VariationalBottleneck should not leak NaNs during rollout with resets."""
        rngs = nnx.Rngs(42)
        obs_size = 16
        latent_size = 8
        action_size = 4
        batch_size = 8
        unroll_length = 20
        max_steps = 5  # Env resets every 5 steps

        # Create mock environment
        env = MockEnv(obs_size, action_size, max_steps=max_steps)

        # Create network with AR1VariationalBottleneck
        # Actor: obs -> hidden -> 2*latent (mean, log_std) -> AR1 -> output -> 2*action
        actor = Sequential(
            [
                Dense(obs_size, 32, rngs, activation=nnx.relu),
                Dense(32, latent_size * 2, rngs, activation=None),
                AR1VariationalBottleneck(
                    latent_size, rngs, kl_weight=0.01, ar1_weight=0.1
                ),
                Dense(latent_size, action_size * 2, rngs, activation=None),
            ]
        )

        # Critic: simple MLP
        critic = Sequential(
            [
                Dense(obs_size, 32, rngs, activation=nnx.relu),
                Dense(32, 1, rngs, activation=None),
            ]
        )

        sampler = NormalTanhSampler(rngs, entropy_weight=1e-3)
        networks = PPOActorCritic(actor=actor, critic=critic, action_sampler=sampler)

        # Initialize states
        key = jax.random.key(0)
        env_keys = jax.random.split(key, batch_size)
        env_states = jax.vmap(env.reset)(env_keys)
        network_states = networks.initialize_state(batch_size)

        # Run rollout
        rollout_key = jax.random.key(1)
        final_net_state, final_env_state, rollout_data = rollout.unroll_env(
            env, env_states, networks, network_states, unroll_length, rollout_key
        )

        # Check that multiple resets occurred
        total_dones = jp.sum(rollout_data.done)
        self.assertGreater(
            float(total_dones), 0, "Expected at least one reset to occur"
        )

        # Check no NaNs in network outputs
        self.assertFalse(
            jp.any(jp.isnan(rollout_data.network_output.actions)),
            "NaN found in actions",
        )
        self.assertFalse(
            jp.any(jp.isnan(rollout_data.network_output.raw_actions)),
            "NaN found in raw_actions",
        )
        self.assertFalse(
            jp.any(jp.isnan(rollout_data.network_output.loglikelihoods)),
            "NaN found in loglikelihoods",
        )
        self.assertFalse(
            jp.any(jp.isnan(rollout_data.network_output.value_estimates)),
            "NaN found in value_estimates",
        )
        self.assertFalse(
            jp.any(jp.isnan(rollout_data.network_output.regularization_loss)),
            "NaN found in regularization_loss",
        )

        # Note: The internal state (last_z) CAN contain NaN for envs that just reset.
        # That's expected behavior - the key test is that NaN doesn't leak to outputs.

    def test_ar1_loss_is_zero_after_reset(self):
        """AR1 loss should be zero immediately after a reset (when prev_z is NaN)."""
        rngs = nnx.Rngs(42)
        latent_size = 8
        batch_size = 4

        ar1 = AR1VariationalBottleneck(
            latent_size, rngs, kl_weight=0.01, ar1_weight=1.0
        )

        # Initialize state (has NaN in last_z)
        state = ar1.initialize_state(batch_size)
        self.assertTrue(jp.all(jp.isnan(state["last_z"])))

        # First forward pass
        x = jp.ones((batch_size, latent_size * 2))
        output = ar1(state, x)

        # l2_diff should be 0.0 because prev_z was NaN
        self.assertTrue(jp.allclose(output.metrics["l2_diff"], jp.zeros(batch_size)))

        # Second forward pass (prev_z is now valid)
        output2 = ar1(output.next_state, x)

        # l2_diff should NOT be zero (unless z happens to equal prev_z exactly)
        # Since we're using fixed input but different RNG, z will differ from prev_z
        # Actually with same input, mean is same but eps differs, so z differs

    def test_ar1_loss_resets_to_zero_on_env_reset(self):
        """AR1 loss should return to zero when reset_state is called."""
        rngs = nnx.Rngs(42)
        latent_size = 8
        batch_size = 4

        ar1 = AR1VariationalBottleneck(
            latent_size, rngs, kl_weight=0.01, ar1_weight=1.0
        )

        # Initialize and run a few steps to get valid prev_z
        state = ar1.initialize_state(batch_size)
        x = jp.ones((batch_size, latent_size * 2))

        for _ in range(3):
            output = ar1(state, x)
            state = output.next_state

        # Now state has valid (non-NaN) last_z
        self.assertFalse(jp.any(jp.isnan(state["last_z"])))

        # Reset the state
        reset_state = ar1.reset_state(state)

        # last_z should be NaN again
        self.assertTrue(jp.all(jp.isnan(reset_state["last_z"])))

        # Next forward pass should have zero l2_diff
        output_after_reset = ar1(reset_state, x)
        self.assertTrue(
            jp.allclose(output_after_reset.metrics["l2_diff"], jp.zeros(batch_size))
        )

    def test_partial_batch_reset(self):
        """Test that partial batch resets work correctly (some envs reset, others don't)."""
        rngs = nnx.Rngs(42)
        latent_size = 8
        batch_size = 4

        ar1 = AR1VariationalBottleneck(
            latent_size, rngs, kl_weight=0.01, ar1_weight=1.0
        )

        # Initialize and run a few steps
        state = ar1.initialize_state(batch_size)
        x = jp.ones((batch_size, latent_size * 2))

        for _ in range(3):
            output = ar1(state, x)
            state = output.next_state

        # Simulate partial reset: envs 0 and 2 reset, envs 1 and 3 don't
        done = jp.array([True, False, True, False])
        reset_state = ar1.reset_state(state)

        # Manually apply partial reset using tree_where
        partial_reset_state = rollout.tree_where(done, reset_state, state)

        # Check that envs 0 and 2 have NaN last_z, envs 1 and 3 don't
        self.assertTrue(jp.all(jp.isnan(partial_reset_state["last_z"][0])))
        self.assertFalse(jp.any(jp.isnan(partial_reset_state["last_z"][1])))
        self.assertTrue(jp.all(jp.isnan(partial_reset_state["last_z"][2])))
        self.assertFalse(jp.any(jp.isnan(partial_reset_state["last_z"][3])))

        # Forward pass
        output = ar1(partial_reset_state, x)

        # l2_diff should be 0 for envs 0 and 2 (just reset), non-zero for 1 and 3
        self.assertEqual(float(output.metrics["l2_diff"][0]), 0.0)
        self.assertEqual(float(output.metrics["l2_diff"][2]), 0.0)
        # Envs 1 and 3 might have non-zero l2_diff (depends on whether z changed)

        # Most importantly: no NaNs in output
        self.assertFalse(jp.any(jp.isnan(output.output)))
        self.assertFalse(jp.any(jp.isnan(output.regularization_loss)))

    def test_no_nan_in_gradients_during_ppo_step(self):
        """AR1VariationalBottleneck should not produce NaN gradients during ppo_step."""
        rngs = nnx.Rngs(42)
        obs_size = 16
        latent_size = 8
        action_size = 4
        n_envs = 8
        rollout_length = 20
        max_steps = 5  # Env resets every 5 steps

        # Create mock environment
        env = MockEnv(obs_size, action_size, max_steps=max_steps)

        # Create network with AR1VariationalBottleneck
        actor = Sequential(
            [
                Dense(obs_size, 32, rngs, activation=nnx.relu),
                Dense(32, latent_size * 2, rngs, activation=None),
                AR1VariationalBottleneck(
                    latent_size, rngs, kl_weight=0.01, ar1_weight=0.1
                ),
                Dense(latent_size, action_size * 2, rngs, activation=None),
            ]
        )

        critic = Sequential(
            [
                Dense(obs_size, 32, rngs, activation=nnx.relu),
                Dense(32, 1, rngs, activation=None),
            ]
        )

        sampler = NormalTanhSampler(rngs, entropy_weight=1e-3)
        networks = PPOActorCritic(actor=actor, critic=critic, action_sampler=sampler)

        # Create training state
        training_state = new_training_state(
            env, networks, n_envs, seed=42, learning_rate=1e-4, gradient_clipping=1.0
        )

        # Run ppo_step
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

        # Check no NaNs in losses (metrics have /mean suffix from _log_metric)
        self.assertFalse(
            jp.any(jp.isnan(metrics["losses/actor/mean"])),
            f"NaN found in actor loss: {metrics['losses/actor/mean']}",
        )
        self.assertFalse(
            jp.any(jp.isnan(metrics["losses/critic/mean"])),
            f"NaN found in critic loss: {metrics['losses/critic/mean']}",
        )
        self.assertFalse(
            jp.any(jp.isnan(metrics["losses/regularization/mean"])),
            f"NaN found in regularization loss: {metrics['losses/regularization/mean']}",
        )

        # Check no NaNs in network parameters after update
        params = nnx.state(new_state.networks, nnx.Param)
        for leaf in jax.tree.leaves(params):
            self.assertFalse(jp.any(jp.isnan(leaf)), f"NaN found in network parameters")


if __name__ == "__main__":
    absltest.main()
