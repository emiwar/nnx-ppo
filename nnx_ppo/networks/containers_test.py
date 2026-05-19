from absl.testing import absltest
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks.containers import Parallel, Splitter, Sequential
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.recurrent import LSTM


class ParallelTest(absltest.TestCase):

    def test_requires_components(self):
        with self.assertRaises(ValueError):
            Parallel()

    def test_output_is_dict_keyed_by_name(self):
        rngs = nnx.Rngs(0)
        p = Parallel(a=Dense(4, 3, rngs), b=Dense(4, 5, rngs))
        state = p.initialize_state(batch_size=2)
        x = jp.ones((2, 4))
        out = p(state, x)
        self.assertEqual(set(out.output.keys()), {"a", "b"})
        self.assertEqual(out.output["a"].shape, (2, 3))
        self.assertEqual(out.output["b"].shape, (2, 5))

    def test_all_components_receive_same_input(self):
        rngs = nnx.Rngs(0)
        # Two identical sub-modules with the same init seed should produce
        # identical outputs from a shared input.
        rngs_a = nnx.Rngs(42)
        rngs_b = nnx.Rngs(42)
        p = Parallel(a=Dense(4, 3, rngs_a), b=Dense(4, 3, rngs_b))
        state = p.initialize_state(batch_size=2)
        x = jp.ones((2, 4))
        out = p(state, x)
        self.assertTrue(jp.allclose(out.output["a"], out.output["b"]))

    def test_carries_per_component_state(self):
        rngs = nnx.Rngs(0)
        p = Parallel(rnn=LSTM(4, 6, rngs), ff=Dense(4, 3, rngs))
        state = p.initialize_state(batch_size=2)
        # State is a dict with one entry per component
        self.assertEqual(set(state.keys()), {"rnn", "ff"})
        # rnn state is (h, c); ff state is ()
        self.assertIsInstance(state["rnn"], tuple)
        self.assertEqual(state["ff"], ())
        # Calling advances rnn state
        x = jp.ones((2, 4))
        out = p(state, x)
        h_old, c_old = state["rnn"]
        h_new, c_new = out.next_state["rnn"]
        self.assertFalse(jp.allclose(h_new, h_old))

    def test_reset_state(self):
        rngs = nnx.Rngs(0)
        p = Parallel(rnn=LSTM(4, 6, rngs), ff=Dense(4, 3, rngs))
        state = p.initialize_state(batch_size=2)
        x = jp.ones((2, 4))
        state = p(state, x).next_state
        reset = p.reset_state(state)
        # rnn state should be zeros after reset
        h, c = reset["rnn"]
        self.assertTrue(jp.allclose(h, 0.0))
        self.assertTrue(jp.allclose(c, 0.0))

    def test_sums_regularization_loss(self):
        # Dense's regularization_loss is a scalar zero; check accumulation works.
        rngs = nnx.Rngs(0)
        p = Parallel(a=Dense(4, 3, rngs), b=Dense(4, 5, rngs))
        state = p.initialize_state(batch_size=2)
        x = jp.ones((2, 4))
        out = p(state, x)
        self.assertTrue(jp.allclose(out.regularization_loss, 0.0))


class SplitterTest(absltest.TestCase):

    def test_requires_at_least_one_slice(self):
        with self.assertRaises(ValueError):
            Splitter()

    def test_rejects_nonpositive_size(self):
        with self.assertRaises(ValueError):
            Splitter(a=0)
        with self.assertRaises(ValueError):
            Splitter(a=-1)

    def test_splits_along_last_axis(self):
        s = Splitter(action=4, value=1)
        x = jp.arange(2 * 5).reshape(2, 5).astype(jp.float32)
        out = s((), x)
        self.assertEqual(out.output["action"].shape, (2, 4))
        self.assertEqual(out.output["value"].shape, (2, 1))
        self.assertTrue(jp.allclose(out.output["action"], x[:, :4]))
        self.assertTrue(jp.allclose(out.output["value"], x[:, 4:5]))

    def test_single_key_relabels(self):
        s = Splitter(action_params=3)
        x = jp.ones((2, 3))
        out = s((), x)
        self.assertEqual(set(out.output.keys()), {"action_params"})
        self.assertTrue(jp.allclose(out.output["action_params"], x))

    def test_preserves_insertion_order(self):
        s = Splitter(c=1, a=2, b=3)
        x = jp.arange(6).reshape(1, 6).astype(jp.float32)
        out = s((), x)
        self.assertTrue(jp.allclose(out.output["c"], x[:, 0:1]))
        self.assertTrue(jp.allclose(out.output["a"], x[:, 1:3]))
        self.assertTrue(jp.allclose(out.output["b"], x[:, 3:6]))

    def test_inside_sequential(self):
        rngs = nnx.Rngs(0)
        action_size = 3
        net = Sequential([
            Dense(4, 8, rngs),
            Dense(8, 2 * action_size + 1, rngs),
            Splitter(action_params=2 * action_size, value=1),
        ])
        state = net.initialize_state(batch_size=2)
        out = net(state, jp.ones((2, 4)))
        self.assertEqual(out.output["action_params"].shape, (2, 2 * action_size))
        self.assertEqual(out.output["value"].shape, (2, 1))


if __name__ == "__main__":
    absltest.main()
