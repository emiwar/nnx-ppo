from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.algorithms.adapter import PPOAdapter
from nnx_ppo.algorithms.distributions import NormalTanhSampler
from nnx_ppo.networks.containers import (
    Parallel,
    PPOActorCritic,
    Sequential,
    Splitter,
)
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.types import Context


def _single_head_adapter(rngs: nnx.Rngs, obs_size: int, action_size: int):
    """Build a single-head PPOAdapter via Splitter + Parallel."""
    # Trunk emits dict {"action_params": ..., "value": ...} matching adapter specs.
    trunk = Parallel(
        action_params=Sequential([
            Dense(obs_size, 8, rngs, activation=nnx.relu),
            Dense(8, 2 * action_size, rngs),
        ]),
        value=Sequential([
            Dense(obs_size, 8, rngs, activation=nnx.relu),
            Dense(8, 1, rngs),
        ]),
    )
    sampler = NormalTanhSampler(rng=rngs, entropy_weight=1e-3)
    return PPOAdapter(
        inner=trunk,
        action_specs={"action_params": sampler},
        value_specs="value",
    )


class PPOAdapterSingleHeadTest(absltest.TestCase):

    def test_constructor_validation(self):
        rngs = nnx.Rngs(0)
        inner = Parallel(action_params=Dense(4, 2, rngs), value=Dense(4, 1, rngs))
        sampler = NormalTanhSampler(rng=rngs, entropy_weight=0.0)
        with self.assertRaises(ValueError):
            PPOAdapter(inner=inner, action_specs={}, value_specs="value")
        with self.assertRaises(ValueError):
            PPOAdapter(inner=inner, action_specs={"action_params": sampler}, value_specs=[])

    def test_forward_pass_shapes(self):
        rngs = nnx.Rngs(42)
        adapter = _single_head_adapter(rngs, obs_size=5, action_size=3)
        state = adapter.initialize_state(batch_size=4)
        obs = jp.ones((4, 5))
        new_state, out = adapter(state, obs, context=Context.ROLLOUT)
        # Single-action mode: output.actions is a raw array, not a dict.
        self.assertEqual(out.actions.shape, (4, 3))
        self.assertEqual(out.raw_actions.shape, (4, 3))
        self.assertEqual(out.loglikelihoods.shape, (4,))
        self.assertEqual(out.value_estimates.shape, (4,))  # squeezed from (4,1)
        # distribution_params has the per-sampler metrics.
        self.assertIn("action_params", out.distribution_params)
        self.assertIn("mu", out.distribution_params["action_params"])
        self.assertIn("sigma", out.distribution_params["action_params"])

    def test_state_initialization(self):
        rngs = nnx.Rngs(0)
        adapter = _single_head_adapter(rngs, obs_size=5, action_size=3)
        state = adapter.initialize_state(batch_size=4)
        self.assertIn("inner", state)
        self.assertIn("samplers", state)
        self.assertIn("action_params", state["samplers"])

    def test_reset_state(self):
        rngs = nnx.Rngs(0)
        adapter = _single_head_adapter(rngs, obs_size=5, action_size=3)
        state = adapter.initialize_state(batch_size=4)
        reset = adapter.reset_state(state)
        # Inner is feedforward (state is list of ()), nothing to verify beyond
        # structure preservation.
        self.assertEqual(set(reset.keys()), {"inner", "samplers"})

    def test_raw_action_threaded_to_sampler(self):
        """When raw_action is provided (single-head mode, raw tensor), the
        sampler should reuse it instead of sampling fresh."""
        rngs = nnx.Rngs(0)
        adapter = _single_head_adapter(rngs, obs_size=5, action_size=3)
        state = adapter.initialize_state(batch_size=2)
        obs = jp.ones((2, 5))
        # First call samples fresh.
        s1, out1 = adapter(state, obs, context=Context.ROLLOUT)
        # Second call with stored raw_action reuses it.
        s2, out2 = adapter(state, obs, raw_action=out1.raw_actions,
                           context=Context.LOSS_REPLAY)
        self.assertTrue(jp.allclose(out2.raw_actions, out1.raw_actions))


class PPOAdapterMultiHeadTest(absltest.TestCase):

    def test_dict_actions_and_values(self):
        rngs = nnx.Rngs(0)
        obs_size = 4
        # Two parallel heads "a" and "b", each producing action_params + value.
        inner = Parallel(
            head_a=Sequential([Dense(obs_size, 5, rngs)]),
            head_b=Sequential([Dense(obs_size, 3, rngs)]),
            value_a=Sequential([Dense(obs_size, 1, rngs)]),
            value_b=Sequential([Dense(obs_size, 1, rngs)]),
        )
        # Each "head_X" emits 2*action_size = 4 (so action_size=2) for "a"
        # and 2 for "b" (so action_size=1). Wait — Dense(obs,5) -> 5 features;
        # for normal-tanh we need even number. Let me use 4 and 2.
        rngs2 = nnx.Rngs(1)
        inner = Parallel(
            head_a=Dense(obs_size, 4, rngs2),     # action_size=2
            head_b=Dense(obs_size, 2, rngs2),     # action_size=1
            value_a=Dense(obs_size, 1, rngs2),
            value_b=Dense(obs_size, 1, rngs2),
        )
        adapter = PPOAdapter(
            inner=inner,
            action_specs={
                "head_a": NormalTanhSampler(rng=rngs2, entropy_weight=0.0),
                "head_b": NormalTanhSampler(rng=rngs2, entropy_weight=0.0),
            },
            value_specs=["value_a", "value_b"],
        )
        state = adapter.initialize_state(batch_size=3)
        obs = jp.ones((3, obs_size))
        _, out = adapter(state, obs, context=Context.ROLLOUT)
        # Multi-head mode: actions / values are dicts.
        self.assertIsInstance(out.actions, dict)
        self.assertEqual(set(out.actions.keys()), {"head_a", "head_b"})
        self.assertEqual(out.actions["head_a"].shape, (3, 2))
        self.assertEqual(out.actions["head_b"].shape, (3, 1))
        self.assertIsInstance(out.value_estimates, dict)
        self.assertEqual(set(out.value_estimates.keys()), {"value_a", "value_b"})
        self.assertEqual(out.value_estimates["value_a"].shape, (3,))


class PPOAdapterLegacyParityTest(absltest.TestCase):
    """An equivalent PPOActorCritic and a PPOAdapter built from the same
    actor/critic/sampler should produce identical outputs for the same input."""

    def test_parity_with_PPOActorCritic(self):
        obs_size = 6
        action_size = 2
        hidden = 8

        # Identical Dense layers via matching seeds.
        rngs_pac_actor = nnx.Rngs(7)
        rngs_pac_critic = nnx.Rngs(8)
        rngs_pac_sampler = nnx.Rngs(9)
        rngs_pa_actor = nnx.Rngs(7)
        rngs_pa_critic = nnx.Rngs(8)
        rngs_pa_sampler = nnx.Rngs(9)

        # PPOActorCritic with no preprocessor.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pac_actor = Sequential([
                Dense(obs_size, hidden, rngs_pac_actor, activation=nnx.relu),
                Dense(hidden, 2 * action_size, rngs_pac_actor),
            ])
            pac_critic = Sequential([
                Dense(obs_size, hidden, rngs_pac_critic, activation=nnx.relu),
                Dense(hidden, 1, rngs_pac_critic),
            ])
            pac_sampler = NormalTanhSampler(rng=rngs_pac_sampler, entropy_weight=1e-3)
            pac = PPOActorCritic(
                actor=pac_actor, critic=pac_critic, action_sampler=pac_sampler
            )

        # Equivalent PPOAdapter.
        pa_actor = Sequential([
            Dense(obs_size, hidden, rngs_pa_actor, activation=nnx.relu),
            Dense(hidden, 2 * action_size, rngs_pa_actor),
        ])
        pa_critic = Sequential([
            Dense(obs_size, hidden, rngs_pa_critic, activation=nnx.relu),
            Dense(hidden, 1, rngs_pa_critic),
        ])
        pa_sampler = NormalTanhSampler(rng=rngs_pa_sampler, entropy_weight=1e-3)
        trunk = Parallel(action_params=pa_actor, value=pa_critic)
        pa = PPOAdapter(
            inner=trunk,
            action_specs={"action_params": pa_sampler},
            value_specs="value",
        )

        # Run both with the same observation in LOSS_REPLAY mode so that the
        # raw_action chosen from PPOActorCritic can be fed to PPOAdapter for a
        # deterministic comparison.
        batch = 4
        obs = jax.random.normal(jax.random.key(0), (batch, obs_size))

        pac_state = pac.initialize_state(batch)
        pa_state = pa.initialize_state(batch)
        _, pac_out = pac(pac_state, obs, context=Context.ROLLOUT)
        _, pa_out = pa(pa_state, obs,
                       raw_action=pac_out.raw_actions,
                       context=Context.LOSS_REPLAY)
        # PAC uses fresh sample from its own RNG; PA replays it via raw_action.
        # Actions and loglikelihoods should match.
        self.assertTrue(
            jp.allclose(pa_out.actions, pac_out.actions, atol=1e-5),
            msg=f"actions mismatch: {pa_out.actions} vs {pac_out.actions}",
        )
        self.assertTrue(
            jp.allclose(pa_out.loglikelihoods, pac_out.loglikelihoods, atol=1e-5)
        )
        self.assertTrue(
            jp.allclose(pa_out.value_estimates, pac_out.value_estimates, atol=1e-5)
        )


if __name__ == "__main__":
    absltest.main()
