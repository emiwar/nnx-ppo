"""Tests for :class:`PopulationGraph`."""

from absl.testing import absltest

import jax.numpy as jp
import numpy as np
from flax import nnx

from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.graph import PopulationGraph
from nnx_ppo.networks.types import Context


class BuildAPITest(absltest.TestCase):

    def test_finalize_required_before_call(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", output_size=3)
        with self.assertRaises(RuntimeError):
            g.initialize_state(1)

    def test_duplicate_population_raises(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", output_size=3)
        with self.assertRaises(ValueError):
            g.add_population("a", output_size=3)

    def test_unknown_endpoint_in_connect(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", output_size=3)
        with self.assertRaises(ValueError):
            g.connect("a", "missing")
        with self.assertRaises(ValueError):
            g.connect("missing", "a")

    def test_default_compute_requires_matching_sizes(self):
        g = PopulationGraph(nnx.Rngs(0))
        with self.assertRaises(ValueError):
            g.add_population("a", output_size=3, input_size=5)

    def test_delay_zero_cycle_is_error(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", output_size=2)
        g.add_population("b", output_size=2)
        g.connect("a", "b")
        g.connect("b", "a")  # delay-0 cycle
        with self.assertRaises(ValueError):
            g.finalize()

    def test_delayed_self_loop_is_ok(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", output_size=2)
        g.connect("a", "a", delay=1)
        g.finalize()  # should not raise
        # And the population should have a buffer of length 1.
        self.assertEqual(g.populations["a"].max_outgoing_delay, 1)

    def test_negative_delay_rejected(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", output_size=2)
        g.add_population("b", output_size=2)
        with self.assertRaises(ValueError):
            g.connect("a", "b", delay=-1)

    def test_cannot_modify_after_finalize(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", output_size=2)
        g.finalize()
        with self.assertRaises(RuntimeError):
            g.add_population("b", output_size=2)
        with self.assertRaises(RuntimeError):
            g.connect("a", "a", delay=1)
        with self.assertRaises(RuntimeError):
            g.add_output("o", source="a")
        with self.assertRaises(RuntimeError):
            g.finalize()


class ForwardPassTest(absltest.TestCase):

    def test_single_input_population_passthrough(self):
        # in_x ---identity--> output `o`
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("in_x", output_size=3, input_from="x")
        g.add_output("o", source="in_x")
        g.finalize()

        state = g.initialize_state(batch_size=2)
        obs = {"x": jp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
        out = g(state, obs, context=Context.INFERENCE)
        np.testing.assert_array_equal(out.output["o"], obs["x"])

    def test_sum_integration_two_inputs(self):
        # in_a + in_b -> hidden (sum). Use identity transforms for clarity.
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("in_a", output_size=2, input_from="a")
        g.add_population("in_b", output_size=2, input_from="b")
        # Hidden is sum of (Dense(in_a) + Dense(in_b)) — replace both
        # default Denses with identity transforms by passing fixed-weight
        # Linears would be heavy; instead use a tiny custom transform.
        from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput

        class Passthrough(StatefulModule):
            def __call__(self, s, x, *, context=Context.INFERENCE):
                return StatefulModuleOutput(s, x, jp.array(0.0), {})

        g.add_population("hidden", output_size=2)
        g.connect("in_a", "hidden", transform=Passthrough())
        g.connect("in_b", "hidden", transform=Passthrough())
        g.add_output("h", source="hidden")
        g.finalize()

        state = g.initialize_state(2)
        obs = {
            "a": jp.array([[1.0, 1.0], [2.0, 2.0]]),
            "b": jp.array([[10.0, 10.0], [20.0, 20.0]]),
        }
        out = g(state, obs)
        np.testing.assert_array_equal(out.output["h"], obs["a"] + obs["b"])

    def test_activation_applied_once_per_population(self):
        # If activation lived on connections, two connections feeding the
        # same destination would apply activation twice and sum after.
        # We assert the opposite: sum first, activate once.
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("in_a", output_size=2, input_from="a")
        g.add_population("in_b", output_size=2, input_from="b")
        from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput

        class Passthrough(StatefulModule):
            def __call__(self, s, x, *, context=Context.INFERENCE):
                return StatefulModuleOutput(s, x, jp.array(0.0), {})

        g.add_population("hidden", output_size=2, activation=jp.tanh)
        g.connect("in_a", "hidden", transform=Passthrough())
        g.connect("in_b", "hidden", transform=Passthrough())
        g.add_output("h", source="hidden")
        g.finalize()

        a = jp.array([[0.5, 0.5]])
        b = jp.array([[0.5, 0.5]])
        out = g(g.initialize_state(1), {"a": a, "b": b})
        np.testing.assert_allclose(out.output["h"], jp.tanh(a + b))

    def test_dense_default_transform_shape(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", output_size=4, input_from="a")
        g.add_population("b", output_size=2)
        g.connect("a", "b")
        g.add_output("o", source="b")
        g.finalize()

        # Default transform should be a Dense(4, 2)
        self.assertEqual(g.connections[0].transform.in_features, 4)
        self.assertEqual(g.connections[0].transform.out_features, 2)

        obs = {"a": jp.zeros((3, 4))}
        out = g(g.initialize_state(3), obs)
        self.assertEqual(out.output["o"].shape, (3, 2))

    def test_recurrent_self_loop_with_delay(self):
        # `a` reads obs[x] and itself (delayed by 1). With identity
        # transforms and no activation, a_t = obs_t + a_{t-1}.
        from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput

        class Passthrough(StatefulModule):
            def __call__(self, s, x, *, context=Context.INFERENCE):
                return StatefulModuleOutput(s, x, jp.array(0.0), {})

        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", output_size=2, input_from="x")
        g.connect("a", "a", delay=1, transform=Passthrough())
        g.add_output("o", source="a")
        g.finalize()

        state = g.initialize_state(1)
        x_seq = [jp.array([[1.0, 0.0]]), jp.array([[2.0, 0.0]]),
                 jp.array([[3.0, 0.0]])]
        outs = []
        for x in x_seq:
            out = g(state, {"x": x})
            outs.append(out.output["o"])
            state = out.next_state
        # Expected cumulative sum: 1, 1+2=3, 1+2+3=6.
        np.testing.assert_allclose(outs[0], jp.array([[1.0, 0.0]]))
        np.testing.assert_allclose(outs[1], jp.array([[3.0, 0.0]]))
        np.testing.assert_allclose(outs[2], jp.array([[6.0, 0.0]]))

    def test_reset_state_zeros_buffer(self):
        from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput

        class Passthrough(StatefulModule):
            def __call__(self, s, x, *, context=Context.INFERENCE):
                return StatefulModuleOutput(s, x, jp.array(0.0), {})

        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", output_size=2, input_from="x")
        g.connect("a", "a", delay=1, transform=Passthrough())
        g.add_output("o", source="a")
        g.finalize()

        state = g.initialize_state(1)
        for x in [jp.array([[1.0, 1.0]]), jp.array([[2.0, 2.0]])]:
            out = g(state, {"x": x})
            state = out.next_state
        # Buffer should be nonzero now; resetting should zero it.
        self.assertGreater(float(jp.abs(state["populations"]["a"]["buffer"]).sum()), 0)
        reset = g.reset_state(state)
        np.testing.assert_array_equal(
            reset["populations"]["a"]["buffer"],
            jp.zeros_like(reset["populations"]["a"]["buffer"]),
        )
        np.testing.assert_array_equal(
            reset["populations"]["a"]["buffer_idx"],
            jp.zeros_like(reset["populations"]["a"]["buffer_idx"]),
        )

    def test_head_applied_to_output(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", output_size=4, input_from="x")
        head = Dense(4, 1, rngs=nnx.Rngs(1))
        g.add_output("v", source="a", head=head)
        g.finalize()

        state = g.initialize_state(2)
        obs = {"x": jp.ones((2, 4))}
        out = g(state, obs)
        self.assertEqual(out.output["v"].shape, (2, 1))

    def test_dense_compute_with_activation(self):
        # Mirrors the nervenet-style input population pattern.
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population(
            "in_x",
            input_size=3,
            output_size=4,
            compute=Dense(3, 4, rngs=nnx.Rngs(2)),
            activation=nnx.swish,
            input_from="x",
        )
        g.add_output("o", source="in_x")
        g.finalize()

        state = g.initialize_state(2)
        obs = {"x": jp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
        out = g(state, obs)
        self.assertEqual(out.output["o"].shape, (2, 4))

    def test_jit_compiles(self):
        g = PopulationGraph(nnx.Rngs(0))
        g.add_population("a", output_size=4, input_from="x")
        g.add_population("b", output_size=2)
        g.connect("a", "b")
        g.connect("b", "b", delay=1)
        g.add_output("o", source="b")
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
