"""Tests for the two-port :class:`PPOAdapter`."""

from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Parallel, Sequential
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.types import (
    PPONetworkOutput,
    StatefulModuleOutput,
)
from nnx_ppo.networks.utils import Filter, Map


def _single_head_adapter(rngs: nnx.Rngs, obs_size: int, action_size: int):
    """Build a single-head PPOAdapter directly (no Parallel wrapper).

    Both ports receive the same upstream obs; the action port runs the
    actor + sampler chain, and the value port runs the critic chain.
    """
    actor = Sequential([
        Dense(obs_size, 8, rngs, activation=nnx.relu),
        Dense(8, 2 * action_size, rngs),
        NormalTanhSampler(rng=rngs, entropy_weight=1e-3),
    ])
    critic = Sequential([
        Dense(obs_size, 8, rngs, activation=nnx.relu),
        Dense(8, 1, rngs),
    ])
    return PPOAdapter(
        action=actor,
        value=critic,
    )


class PPOAdapterSingleHeadTest(absltest.TestCase):

    def test_forward_pass_shapes(self):
        rngs = nnx.Rngs(42)
        net = _single_head_adapter(rngs, obs_size=5, action_size=3)
        state = net.initialize_state(batch_size=4)
        obs = jp.ones((4, 5))
        out = net(state, obs)
        self.assertIsInstance(out, StatefulModuleOutput)
        self.assertIsInstance(out.output, PPONetworkOutput)
        self.assertEqual(out.output.actions.shape, (4, 3))
        self.assertEqual(out.output.loglikelihoods.shape, (4,))
        self.assertEqual(out.output.value_estimates.shape, (4,))  # squeezed (4,1)

    def test_state_structure(self):
        rngs = nnx.Rngs(0)
        net = _single_head_adapter(rngs, obs_size=5, action_size=3)
        state = net.initialize_state(batch_size=4)
        self.assertIn("action", state)
        self.assertIn("value", state)

    def test_rollout_extras_emitted_and_consumed(self):
        rngs = nnx.Rngs(0)
        net = _single_head_adapter(rngs, obs_size=5, action_size=3)
        state = net.initialize_state(batch_size=2)
        obs = jp.ones((2, 5))
        # Sampling pass: rollout_extras=None → sample fresh, emit raw action.
        out1 = net(state, obs)
        self.assertIsNotNone(out1.rollout_extras)
        # Replay pass: feed the stored extras back in to reproduce the
        # action and log-likelihood under the same weights.
        out2 = net(state, obs, out1.rollout_extras)
        self.assertTrue(jp.allclose(out1.output.actions, out2.output.actions))
        self.assertTrue(
            jp.allclose(out1.output.loglikelihoods, out2.output.loglikelihoods)
        )


class PPOAdapterMultiHeadTest(absltest.TestCase):

    def test_per_key_actions_and_values(self):
        rngs = nnx.Rngs(0)
        obs_size = 4
        pops = ("a", "b")
        # Trunk emits {"action_a": ..., "action_b": ..., "value_a": ..., "value_b": ...}.
        components = {
            "action_a": Dense(obs_size, 4, rngs),  # action_size=2
            "action_b": Dense(obs_size, 2, rngs),  # action_size=1
            "value_a": Dense(obs_size, 1, rngs),
            "value_b": Dense(obs_size, 1, rngs),
        }
        trunk = Parallel(components)
        # Action port: per-key dispatch via Map.
        action_port = Sequential([
            Filter({p: f"action_{p}" for p in pops}),
            Map({p: NormalTanhSampler(rng=rngs, entropy_weight=0.0) for p in pops}),
        ])
        value_port = Filter({p: f"value_{p}" for p in pops})
        net = Sequential([trunk, PPOAdapter(action=action_port, value=value_port)])

        state = net.initialize_state(batch_size=3)
        obs = jp.ones((3, obs_size))
        out = net(state, obs)
        self.assertIsInstance(out.output, PPONetworkOutput)
        self.assertEqual(set(out.output.actions.keys()), set(pops))
        self.assertEqual(out.output.actions["a"].shape, (3, 2))
        self.assertEqual(out.output.actions["b"].shape, (3, 1))
        self.assertEqual(set(out.output.value_estimates.keys()), set(pops))
        self.assertEqual(out.output.value_estimates["a"].shape, (3,))


if __name__ == "__main__":
    absltest.main()
