"""Tests for activation recording (:mod:`nnx_ppo.networks.recording`)."""

from absl.testing import absltest

import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.containers import Parallel, Sequential
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.graph import PopulationGraph
from nnx_ppo.networks.recurrent import LSTM
from nnx_ppo.networks.recording import (
    ACTIVATION_KEY,
    Recorder,
    _is_leaf,
    extract_activations,
    with_recording,
)


def _example_net() -> Sequential:
    rngs = nnx.Rngs(0)
    return Sequential(
        [
            Dense(4, 8, rngs, activation=nnx.relu),
            Parallel(rnn=LSTM(8, 6, rngs), ff=Dense(8, 3, rngs)),
        ]
    )


class IsLeafTest(absltest.TestCase):

    def test_leaf_and_container_classification(self):
        net = _example_net()
        self.assertTrue(_is_leaf(net.layers[0]))  # Dense
        self.assertTrue(_is_leaf(net.layers[1].components["ff"]))  # Dense
        self.assertTrue(_is_leaf(net.layers[1].components["rnn"]))  # LSTM
        self.assertFalse(_is_leaf(net))  # Sequential
        self.assertFalse(_is_leaf(net.layers[1]))  # Parallel

    def test_non_module_is_not_leaf(self):
        self.assertFalse(_is_leaf(jp.ones(3)))


class RecorderTest(absltest.TestCase):

    def test_passes_output_and_state_through_and_records(self):
        rngs = nnx.Rngs(0)
        dense = Dense(4, 5, rngs)
        rec = Recorder(dense)
        state = rec.initialize_state(2)
        x = jp.ones((2, 4))

        ref = dense(state, x)
        out = rec(state, x)

        self.assertTrue(jp.allclose(out.output, ref.output))
        self.assertEqual(jax.tree.structure(out.next_state),
                         jax.tree.structure(ref.next_state))
        # Activation recorded under the reserved key and equals the output.
        self.assertIn(ACTIVATION_KEY, out.metrics)
        self.assertTrue(jp.allclose(out.metrics[ACTIVATION_KEY], ref.output))

    def test_delegates_state_lifecycle(self):
        rngs = nnx.Rngs(0)
        rec = Recorder(LSTM(4, 6, rngs))
        state = rec.initialize_state(2)
        # LSTM carry is (h, c), both [batch, hidden].
        self.assertEqual(state[0].shape, (2, 6))
        self.assertEqual(state[1].shape, (2, 6))
        reset = rec.reset_state(state)
        self.assertEqual(reset[0].shape, (2, 6))


class WithRecordingTest(absltest.TestCase):

    def test_wraps_leaves_keeps_containers(self):
        net = _example_net()
        rec = with_recording(net)
        self.assertIsInstance(rec.layers[0], Recorder)
        self.assertEqual(type(rec.layers[1]).__name__, "Parallel")
        self.assertIsInstance(rec.layers[1].components["rnn"], Recorder)
        self.assertIsInstance(rec.layers[1].components["ff"], Recorder)

    def test_no_double_wrapping(self):
        net = _example_net()
        rec = with_recording(net)
        # The wrapped module is the original leaf, not another Recorder.
        self.assertNotIsInstance(rec.layers[0].wrapped, Recorder)
        self.assertIsInstance(rec.layers[0].wrapped, Dense)

    def test_original_network_untouched(self):
        net = _example_net()
        with_recording(net)
        self.assertNotIsInstance(net.layers[0], Recorder)

    def test_double_call_warns_and_returns_unchanged(self):
        rec = with_recording(_example_net())
        with self.assertWarns(UserWarning):
            again = with_recording(rec)
        self.assertIs(again, rec)
        # No second layer of wrapping was introduced.
        self.assertIsInstance(again.layers[0].wrapped, Dense)

    def test_forward_output_identical(self):
        net = _example_net()
        rec = with_recording(net)
        state = net.initialize_state(2)
        x = jp.ones((2, 4))
        out_ref = net(state, x)
        out_rec = rec(state, x)
        identical = jax.tree.all(
            jax.tree.map(lambda a, b: bool(jp.allclose(a, b)),
                         out_ref.output, out_rec.output)
        )
        self.assertTrue(identical)

    def test_extract_activations_keyed_by_path(self):
        net = _example_net()
        rec = with_recording(net)
        state = net.initialize_state(2)
        out = rec(state, jp.ones((2, 4)))
        acts = extract_activations(out.metrics)
        # Sequential -> positional int keys; Parallel -> string keys.
        self.assertEqual(acts[0].shape, (2, 8))          # Dense
        self.assertEqual(acts[1]["ff"].shape, (2, 3))    # Parallel/ff
        self.assertEqual(acts[1]["rnn"].shape, (2, 6))   # Parallel/rnn


class ExtractActivationsTest(absltest.TestCase):

    def test_drops_real_metrics_and_empty_branches(self):
        a0 = jp.ones((2, 3))
        a1 = jp.ones((2, 4))
        metrics = {
            0: {ACTIVATION_KEY: a0, "entropy": jp.array(1.0)},  # leaf + real metric
            1: {"inner": {ACTIVATION_KEY: a1}},                 # nested container
            2: {"only_scalar": jp.array(0.5)},                  # no activation
        }
        acts = extract_activations(metrics)
        self.assertEqual(set(acts.keys()), {0, 1})
        self.assertTrue(jp.allclose(acts[0], a0))
        self.assertTrue(jp.allclose(acts[1]["inner"], a1))

    def test_returns_none_without_activations(self):
        self.assertIsNone(extract_activations({"a": jp.array(1.0)}))
        self.assertIsNone(extract_activations(jp.array(1.0)))


class GraphRecordingTest(absltest.TestCase):

    def _graph(self) -> PopulationGraph:
        g = PopulationGraph(nnx.Rngs(0))
        g.add_input("in", size=4, input_from="in")
        g.add_population("hidden", size=5, activation=nnx.tanh)
        g.add_output("mot", size=3, output_to="mot")
        g.connect("in", "hidden")
        g.connect("hidden", "mot")
        g.finalize()
        return g

    def test_flag_enabled_on_copy_only(self):
        g = self._graph()
        rec = with_recording(g)
        self.assertTrue(rec.record_activations)
        self.assertFalse(g.record_activations)

    def test_emits_one_activation_per_population(self):
        rec = with_recording(self._graph())
        state = rec.initialize_state(2)
        out = rec(state, {"in": jp.ones((2, 4))})
        acts = extract_activations(out.metrics)
        self.assertEqual(acts["in"].shape, (2, 4))
        self.assertEqual(acts["hidden"].shape, (2, 5))
        self.assertEqual(acts["mot"].shape, (2, 3))

    def test_no_recording_when_flag_off(self):
        g = self._graph()
        state = g.initialize_state(2)
        out = g(state, {"in": jp.ones((2, 4))})
        self.assertEqual(out.metrics, {})


if __name__ == "__main__":
    absltest.main()
