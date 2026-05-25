"""Tests for :class:`PopulationGraph`."""

from absl.testing import absltest

import jax.numpy as jp
import numpy as np
from flax import nnx

from nnx_ppo.networks.graph import PopulationGraph
from nnx_ppo.networks.types import (
    Context,
    StatefulModule,
    StatefulModuleOutput,
)


class _Passthrough(StatefulModule):
    """Identity transform used as a custom (non-Dense) connection
    transform in tests so we can assert on raw arithmetic without the
    learned-Dense complication."""

    def __call__(self, s, x, *, context=Context.INFERENCE):
        return StatefulModuleOutput(s, x, jp.array(0.0), {})


class BuildAPITest(absltest.TestCase):

    def test_finalize_required_before_call(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", size=3)
        with self.assertRaises(RuntimeError):
            g.initialize_state(1)

    def test_duplicate_population_raises(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", size=3)
        with self.assertRaises(ValueError):
            g.add_population("a", size=3)

    def test_unknown_endpoint_in_connect(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", size=3)
        with self.assertRaises(ValueError):
            g.connect("a", "missing")
        with self.assertRaises(ValueError):
            g.connect("missing", "a")

    def test_delay_zero_cycle_is_error(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", size=2)
        g.add_population("b", size=2)
        g.connect("a", "b")
        g.connect("b", "a")  # delay-0 cycle
        with self.assertRaises(ValueError):
            g.finalize()

    def test_delayed_self_loop_is_ok(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", size=2)
        g.connect("a", "a", delay=1)
        g.finalize()  # should not raise
        self.assertEqual(g.populations["a"].max_outgoing_delay, 1)

    def test_negative_delay_rejected(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", size=2)
        g.add_population("b", size=2)
        with self.assertRaises(ValueError):
            g.connect("a", "b", delay=-1)

    def test_reciprocal_adds_two_connections(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", size=3)
        g.add_population("b", size=3)
        g.connect("a", "b", delay=1, reciprocal=True)
        g.finalize()
        self.assertEqual(len(g.connections), 2)
        self.assertEqual(g.connections[0].src, "a")
        self.assertEqual(g.connections[0].dst, "b")
        self.assertEqual(g.connections[1].src, "b")
        self.assertEqual(g.connections[1].dst, "a")

    def test_reciprocal_rejects_custom_transform(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", size=2)
        g.add_population("b", size=2)
        with self.assertRaises(ValueError):
            g.connect("a", "b", transform=_Passthrough(), reciprocal=True)

    def test_cannot_modify_after_finalize(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", size=2)
        g.finalize()
        with self.assertRaises(RuntimeError):
            g.add_population("b", size=2)
        with self.assertRaises(RuntimeError):
            g.connect("a", "a", delay=1)
        with self.assertRaises(RuntimeError):
            g.add_input("c", size=2, input_from="c")
        with self.assertRaises(RuntimeError):
            g.add_output("o", size=2)
        with self.assertRaises(RuntimeError):
            g.finalize()


class ForwardPassTest(absltest.TestCase):

    def test_input_population_passthrough(self):
        # An input pop that also has output_to=key emits its obs entry
        # back out unchanged (no activation, no incoming connections).
        g = PopulationGraph(nnx.Rngs(0))
        g.add_input("in_x", size=3, input_from="x")
        g.add_output("o", size=3)
        g.connect("in_x", "o", transform=_Passthrough())
        g.finalize()

        state = g.initialize_state(batch_size=2)
        obs = {"x": jp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
        out = g(state, obs, context=Context.INFERENCE)
        np.testing.assert_array_equal(out.output["o"], obs["x"])

    def test_sum_integration_two_inputs(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_input("in_a", size=2, input_from="a")
        g.add_input("in_b", size=2, input_from="b")
        g.add_population("hidden", size=2)
        g.connect("in_a", "hidden", transform=_Passthrough())
        g.connect("in_b", "hidden", transform=_Passthrough())
        g.add_output("h", size=2)
        g.connect("hidden", "h", transform=_Passthrough())
        g.finalize()

        state = g.initialize_state(2)
        obs = {
            "a": jp.array([[1.0, 1.0], [2.0, 2.0]]),
            "b": jp.array([[10.0, 10.0], [20.0, 20.0]]),
        }
        out = g(state, obs)
        np.testing.assert_array_equal(out.output["h"], obs["a"] + obs["b"])

    def test_activation_applied_once_per_population(self):
        # Two connections feeding a population with tanh activation: the
        # sum is taken first, then activation is applied once.
        g = PopulationGraph(nnx.Rngs(0))
        g.add_input("in_a", size=2, input_from="a")
        g.add_input("in_b", size=2, input_from="b")
        g.add_population("hidden", size=2, activation=jp.tanh)
        g.connect("in_a", "hidden", transform=_Passthrough())
        g.connect("in_b", "hidden", transform=_Passthrough())
        g.add_output("h", size=2)
        g.connect("hidden", "h", transform=_Passthrough())
        g.finalize()

        a = jp.array([[0.5, 0.5]])
        b = jp.array([[0.5, 0.5]])
        out = g(g.initialize_state(1), {"a": a, "b": b})
        np.testing.assert_allclose(out.output["h"], jp.tanh(a + b))

    def test_dense_default_transform_shape(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_input("a", size=4, input_from="a")
        g.add_output("b", size=2)
        g.connect("a", "b")
        g.finalize()

        # Default transform should be Dense(4, 2).
        self.assertEqual(g.connections[0].transform.in_features, 4)
        self.assertEqual(g.connections[0].transform.out_features, 2)

        obs = {"a": jp.zeros((3, 4))}
        out = g(g.initialize_state(3), obs)
        self.assertEqual(out.output["b"].shape, (3, 2))

    def test_recurrent_self_loop_with_delay(self):
        # `a` reads obs[x] and itself (delayed by 1). With identity
        # transforms and no activation, a_t = obs_t + a_{t-1}.
        g = PopulationGraph(nnx.Rngs(0))
        g.add_input("a", size=2, input_from="x")
        g.connect("a", "a", delay=1, transform=_Passthrough())
        # Expose `a` itself; use a passthrough connection so the output
        # equals `a`'s activation.
        g.add_output("o", size=2)
        g.connect("a", "o", transform=_Passthrough())
        g.finalize()

        state = g.initialize_state(1)
        x_seq = [
            jp.array([[1.0, 0.0]]),
            jp.array([[2.0, 0.0]]),
            jp.array([[3.0, 0.0]]),
        ]
        outs = []
        for x in x_seq:
            out = g(state, {"x": x})
            outs.append(out.output["o"])
            state = out.next_state
        # Cumulative sum: 1, 1+2=3, 1+2+3=6.
        np.testing.assert_allclose(outs[0], jp.array([[1.0, 0.0]]))
        np.testing.assert_allclose(outs[1], jp.array([[3.0, 0.0]]))
        np.testing.assert_allclose(outs[2], jp.array([[6.0, 0.0]]))

    def test_reset_state_zeros_buffer(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_input("a", size=2, input_from="x")
        g.connect("a", "a", delay=1, transform=_Passthrough())
        g.add_output("o", size=2)
        g.connect("a", "o", transform=_Passthrough())
        g.finalize()

        state = g.initialize_state(1)
        for x in [jp.array([[1.0, 1.0]]), jp.array([[2.0, 2.0]])]:
            out = g(state, {"x": x})
            state = out.next_state
        self.assertGreater(
            float(jp.abs(state["populations"]["a"]["buffer"]).sum()), 0
        )
        reset = g.reset_state(state)
        np.testing.assert_array_equal(
            reset["populations"]["a"]["buffer"],
            jp.zeros_like(reset["populations"]["a"]["buffer"]),
        )
        np.testing.assert_array_equal(
            reset["populations"]["a"]["buffer_idx"],
            jp.zeros_like(reset["populations"]["a"]["buffer_idx"]),
        )

    def test_output_to_renames_in_output_dict(self):
        # add_output(name="value_arm", output_to="arm") → emit under "arm".
        g = PopulationGraph(nnx.Rngs(0))
        g.add_input("in_x", size=4, input_from="x")
        g.add_output("value_arm", size=4, output_to="arm")
        g.connect("in_x", "value_arm", transform=_Passthrough())
        g.finalize()

        obs = {"x": jp.full((2, 4), 7.0)}
        out = g(g.initialize_state(2), obs)
        self.assertIn("arm", out.output)
        self.assertNotIn("value_arm", out.output)

    def test_output_falls_back_to_name(self):
        # add_output without output_to emits under the population name.
        g = PopulationGraph(nnx.Rngs(0))
        g.add_input("in_x", size=4, input_from="x")
        g.add_output("o", size=4)
        g.connect("in_x", "o", transform=_Passthrough())
        g.finalize()

        out = g(g.initialize_state(2), {"x": jp.ones((2, 4))})
        self.assertIn("o", out.output)

    def test_jit_compiles(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_input("a", size=4, input_from="x")
        g.add_population("b", size=2)
        g.connect("a", "b")
        g.connect("b", "b", delay=1)
        g.add_output("o", size=2)
        g.connect("b", "o", transform=_Passthrough())
        g.finalize()

        @nnx.jit
        def step(graph, state, obs):
            return graph(state, obs)

        state = g.initialize_state(3)
        obs = {"x": jp.ones((3, 4))}
        out = step(g, state, obs)
        self.assertEqual(out.output["o"].shape, (3, 2))


if __name__ == "__main__":
    absltest.main()
