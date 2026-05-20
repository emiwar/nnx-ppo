"""Tests for nnx_ppo.networks.utils."""

from absl.testing import absltest

import jax.numpy as jp
import numpy as np
from flax import nnx

from nnx_ppo.networks.types import (
    Context,
    StatefulModule,
    StatefulModuleOutput,
)
from nnx_ppo.networks.utils import Filter, Flattener, Merge, Scale


class ScaleTest(absltest.TestCase):

    def test_scale_array(self):
        s = Scale(2.5)
        out = s((), jp.ones((3, 4)))
        np.testing.assert_allclose(out.output, jp.full((3, 4), 2.5))

    def test_scale_pytree(self):
        s = Scale(0.5)
        out = s((), {"a": jp.ones((2, 3)), "b": jp.full((2, 1), 4.0)})
        np.testing.assert_allclose(out.output["a"], jp.full((2, 3), 0.5))
        np.testing.assert_allclose(out.output["b"], jp.full((2, 1), 2.0))


class FilterTest(absltest.TestCase):

    def test_string_spec_picks_top_level_key(self):
        f = Filter({"out": "a"})
        out = f((), {"a": jp.ones((2, 3)), "b": jp.zeros((2, 4))})
        np.testing.assert_array_equal(out.output["out"], jp.ones((2, 3)))
        self.assertNotIn("b", out.output)

    def test_tuple_spec_walks_nested_path(self):
        f = Filter({"v": ("arm", "proprio")})
        obs = {"arm": {"proprio": jp.ones((2, 5)), "target": jp.zeros((2, 5))}}
        out = f((), obs)
        np.testing.assert_array_equal(out.output["v"], jp.ones((2, 5)))

    def test_callable_spec_gets_full_input(self):
        f = Filter({"sum": lambda obs: obs["a"] + obs["b"]})
        out = f((), {"a": jp.ones((2, 3)), "b": jp.full((2, 3), 2.0)})
        np.testing.assert_allclose(out.output["sum"], jp.full((2, 3), 3.0))

    def test_mixed_specs(self):
        f = Filter({
            "p": ("arm", "proprio"),
            "z": "head",
            "calc": lambda obs: obs["head"] * 2,
        })
        obs = {
            "arm": {"proprio": jp.ones((2, 4))},
            "head": jp.full((2, 3), 5.0),
        }
        out = f((), obs)
        np.testing.assert_array_equal(out.output["p"], jp.ones((2, 4)))
        np.testing.assert_array_equal(out.output["z"], jp.full((2, 3), 5.0))
        np.testing.assert_allclose(out.output["calc"], jp.full((2, 3), 10.0))

    def test_invalid_spec_type(self):
        with self.assertRaises(TypeError):
            Filter({"x": 5})  # int spec — not allowed

    def test_non_dict_spec(self):
        with self.assertRaises(TypeError):
            Filter([("a", "b")])  # list — must be dict

    def test_reveal_targets_pattern(self):
        # Mirrors the v3 reveal_targets="joystick_only" use case.
        obs = {
            "arm_L": {"proprioception": jp.ones((2, 3)), "target": jp.zeros((2, 4))},
            "leg_L": {"proprioception": jp.full((2, 5), 2.0), "target": jp.zeros((2, 4))},
            "root": {"future_target": {"pos": jp.full((2, 3), 7.0)}},
        }
        f = Filter({
            "arm_L": ("arm_L", "proprioception"),
            "leg_L": ("leg_L", "proprioception"),
            "root": ("root", "future_target", "pos"),
        })
        out = f((), obs)
        self.assertEqual(set(out.output.keys()), {"arm_L", "leg_L", "root"})
        np.testing.assert_array_equal(out.output["root"], jp.full((2, 3), 7.0))


class FlattenerTest(absltest.TestCase):

    def test_depth_zero_default_flatten(self):
        f = Flattener()
        x = {"a": jp.ones((2, 3)), "b": jp.full((2, 5), 4.0)}
        out = f((), x)
        self.assertEqual(out.output.shape, (2, 8))

    def test_depth_zero_already_flat(self):
        f = Flattener()
        out = f((), jp.ones((2, 7)))
        self.assertEqual(out.output.shape, (2, 7))

    def test_preserve_one_level(self):
        f = Flattener(preserve_levels=1)
        x = {
            "arm": {"proprio": jp.ones((2, 4)), "target": jp.zeros((2, 8))},
            "root": jp.full((2, 6), 3.0),
        }
        out = f((), x)
        self.assertIsInstance(out.output, dict)
        self.assertEqual(set(out.output.keys()), {"arm", "root"})
        self.assertEqual(out.output["arm"].shape, (2, 12))
        self.assertEqual(out.output["root"].shape, (2, 6))

    def test_preserve_one_level_idempotent_on_flat_values(self):
        f = Flattener(preserve_levels=1)
        x = {"a": jp.ones((3, 4)), "b": jp.zeros((3, 2))}
        out = f((), x)
        np.testing.assert_array_equal(out.output["a"], x["a"])
        np.testing.assert_array_equal(out.output["b"], x["b"])

    def test_preserve_two_levels(self):
        f = Flattener(preserve_levels=2)
        x = {
            "arm": {
                "p": {"a": jp.ones((2, 3)), "b": jp.zeros((2, 4))},
                "t": jp.full((2, 5), 1.0),
            },
        }
        out = f((), x)
        self.assertEqual(out.output["arm"]["p"].shape, (2, 7))
        self.assertEqual(out.output["arm"]["t"].shape, (2, 5))

    def test_preserve_negative_rejected(self):
        with self.assertRaises(ValueError):
            Flattener(preserve_levels=-1)

    def test_preserve_too_deep_raises(self):
        # preserve_levels=1 but the value at depth 1 is a leaf — error.
        f = Flattener(preserve_levels=2)
        with self.assertRaises(TypeError):
            f((), {"a": jp.ones((2, 3))})


class MergeTest(absltest.TestCase):

    class _DictEmittingModule(StatefulModule):
        def __init__(self, keys):
            self._keys = tuple(keys)

        def __call__(self, state, x, *, context: Context = Context.INFERENCE):
            return StatefulModuleOutput(
                state, {k: x for k in self._keys}, jp.array(0.0), {}
            )

    def test_basic_merge(self):
        m = Merge(
            a=MergeTest._DictEmittingModule(["motor_arm", "motor_leg"]),
            b=MergeTest._DictEmittingModule(["value"]),
        )
        state = m.initialize_state(2)
        out = m(state, jp.ones((2, 3)))
        self.assertEqual(
            set(out.output.keys()), {"motor_arm", "motor_leg", "value"}
        )

    def test_duplicate_key_raises(self):
        m = Merge(
            a=MergeTest._DictEmittingModule(["k"]),
            b=MergeTest._DictEmittingModule(["k"]),
        )
        state = m.initialize_state(2)
        with self.assertRaises(ValueError):
            m(state, jp.ones((2, 3)))

    def test_non_dict_component_raises(self):
        class _ArrayModule(StatefulModule):
            def __call__(self, state, x, *, context: Context = Context.INFERENCE):
                return StatefulModuleOutput(state, x, jp.array(0.0), {})

        m = Merge(bad=_ArrayModule())
        state = m.initialize_state(2)
        with self.assertRaises(TypeError):
            m(state, jp.ones((2, 3)))

    def test_empty_merge_rejected(self):
        with self.assertRaises(ValueError):
            Merge()

    def test_state_threading(self):
        # Each component sees its own carry slot.
        class _CountingModule(StatefulModule):
            def __call__(self, state, x, *, context: Context = Context.INFERENCE):
                return StatefulModuleOutput(
                    state + 1, {self._key: x}, jp.array(0.0), {}
                )

            def __init__(self, key):
                self._key = key

            def initialize_state(self, batch_size):
                return jp.zeros(batch_size, jp.int32)

        m = Merge(a=_CountingModule("ka"), b=_CountingModule("kb"))
        state = m.initialize_state(4)
        out = m(state, jp.ones((4, 2)))
        np.testing.assert_array_equal(out.next_state["a"], jp.ones(4, jp.int32))
        np.testing.assert_array_equal(out.next_state["b"], jp.ones(4, jp.int32))


class ComposesWithPPOAdapterTest(absltest.TestCase):
    """End-to-end: Flattener(preserve_levels=1) + Filter + Merge feeding PPOAdapter."""

    def test_merge_feeds_adapter_flat_dict(self):
        from nnx_ppo.algorithms.adapter import PPOAdapter
        from nnx_ppo.algorithms.distributions import NormalTanhSampler
        from nnx_ppo.networks.containers import Sequential
        from nnx_ppo.networks.feedforward import Dense

        rngs = nnx.Rngs(0, action_sampling=1)

        # Two independent stacks, each emitting a dict of named heads.
        actor_stack = Sequential([
            Flattener(),
            Dense(7, 4, rngs),
        ])

        class _RenameAsAction(StatefulModule):
            def __call__(self, s, x, *, context=Context.INFERENCE):
                return StatefulModuleOutput(s, {"action_params": x}, jp.array(0.0), {})

        class _RenameAsValue(StatefulModule):
            def __call__(self, s, x, *, context=Context.INFERENCE):
                return StatefulModuleOutput(s, {"value": x}, jp.array(0.0), {})

        actor_branch = Sequential([actor_stack, _RenameAsAction()])
        critic_branch = Sequential([
            Flattener(),
            Dense(7, 1, rngs),
            _RenameAsValue(),
        ])
        trunk = Merge(actor=actor_branch, critic=critic_branch)

        nets = PPOAdapter(
            inner=trunk,
            action_specs={"action_params": NormalTanhSampler(rngs, entropy_weight=1e-2)},
            value_specs="value",
        )
        state = nets.initialize_state(3)
        obs = {"x": jp.ones((3, 4)), "y": jp.ones((3, 3))}
        state, out = nets(state, obs)
        self.assertEqual(out.actions.shape, (3, 2))
        self.assertEqual(out.value_estimates.shape, (3,))


if __name__ == "__main__":
    absltest.main()
