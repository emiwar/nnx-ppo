from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks import factories, types
from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Sequential
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.sampling_layers import NormalTanhSampler


def _find(net: Sequential, cls):
    for layer in net.layers:
        if isinstance(layer, cls):
            return layer
    return None


class MakeMLPActorCriticTest(absltest.TestCase):

    def setUp(self):
        SEED = 12345
        self.obs_size = 10
        self.action_size = 5
        self.hidden_sizes = [17, 39]
        rngs = nnx.Rngs(SEED)
        self.mlp_net = factories.make_mlp_actor_critic(
            self.obs_size, self.action_size, self.hidden_sizes, self.hidden_sizes, rngs
        )

    def test_factory_returns_sequential(self):
        self.assertIsInstance(self.mlp_net, Sequential)

    def test_actor_and_critic_independently_initialized(self):
        adapter = _find(self.mlp_net, PPOAdapter)
        assert isinstance(adapter, PPOAdapter)
        # Action port: Sequential([*actor_layers, sampler]); the actor Dense
        # layers are everything except the trailing sampler.
        action_layers = list(adapter.action.layers)
        actor_denses = [layer for layer in action_layers if isinstance(layer, Dense)]
        critic_denses = [layer for layer in adapter.value.layers if isinstance(layer, Dense)]
        self.assertGreater(len(actor_denses), 0)
        self.assertEqual(len(actor_denses), len(critic_denses))
        for actor_dense, critic_dense in zip(actor_denses, critic_denses):
            self.assertFalse(
                jp.allclose(
                    actor_dense.linear.kernel[...], critic_dense.linear.kernel[...]
                )
            )

    def test_forward_pass_shape(self):
        n_envs = 256
        state = self.mlp_net.initialize_state(n_envs)
        key = jax.random.key(seed=21)
        obs = jax.random.normal(key, (n_envs, self.obs_size))
        out = self.mlp_net(state, obs)
        self.assertIsInstance(out.output, types.PPONetworkOutput)
        self.assertEqual(out.output.actions.shape, (n_envs, self.action_size))
        self.assertEqual(out.output.loglikelihoods.shape, (n_envs,))
        self.assertEqual(out.output.value_estimates.shape, (n_envs,))

    def test_sampler_train_and_eval_mode(self):
        adapter = _find(self.mlp_net, PPOAdapter)
        assert isinstance(adapter, PPOAdapter)
        # action port: Sequential([actor, sampler])
        sampler = adapter.action.layers[-1]
        assert isinstance(sampler, NormalTanhSampler)
        self.assertFalse(sampler.deterministic)
        self.mlp_net.eval()
        self.assertTrue(sampler.deterministic)
        self.mlp_net.train()
        self.assertFalse(sampler.deterministic)

    def test_normalize_obs_update_statistics(self):
        SEED = 42
        OBS_SIZE = 24
        ACTION_SIZE = 5
        BATCH_SIZE = 32
        N_STEPS = 4
        nets = factories.make_mlp_actor_critic(
            OBS_SIZE,
            ACTION_SIZE,
            actor_hidden_sizes=[64, 64],
            critic_hidden_sizes=[64, 64],
            rngs=nnx.Rngs(SEED, action_sampling=SEED),
            normalize_obs=True,
        )
        normalizer = _find(nets, Normalizer)
        assert isinstance(normalizer, Normalizer)

        key = jax.random.key(SEED)
        mean_key, var_key = jax.random.split(key)
        data = jax.random.normal(mean_key, (OBS_SIZE,)) + jax.random.normal(
            var_key, (N_STEPS, BATCH_SIZE, OBS_SIZE)
        )

        # Run ROLLOUT forward over T steps, collecting the per-step
        # rollout_extras the Normalizer emits, then feed the [T, B, ...]
        # history into update_statistics.
        state = nets.initialize_state(BATCH_SIZE)
        per_step_extras = []
        for t in range(N_STEPS):
            out = nets(state, data[t])
            per_step_extras.append(out.rollout_extras)
            state = out.next_state
        rollout_extras = jax.tree.map(
            lambda *xs: jp.stack(xs, axis=0), *per_step_extras
        )
        nets.update_statistics(rollout_extras)

        true_mean = jp.mean(data, axis=(0, 1))
        true_std = jp.std(data, axis=(0, 1))
        est_mean = normalizer.mean[...]
        est_std = jp.sqrt(normalizer.M2[...] / normalizer.counter[...])
        self.assertEqual(int(normalizer.counter[...]), N_STEPS * BATCH_SIZE)
        self.assertLess(float(jp.max(jp.abs(est_mean - true_mean))), 1e-5)
        self.assertLess(float(jp.max(jp.abs(est_std - true_std))), 1e-5)


if __name__ == "__main__":
    absltest.main()
