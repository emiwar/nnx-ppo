from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx

from nnx_ppo.networks import modules, types

class ModulesTest(absltest.TestCase):

    def setUp(self):
        SEED = 12345
        self.obs_size = 10
        self.action_size = 5
        self.hidden_sizes = [17, 39]
        rngs = nnx.Rngs(SEED)
        self.mlp_net = modules.MLPActorCritic(self.obs_size,
                                              self.action_size,
                                              self.hidden_sizes,
                                              self.hidden_sizes,
                                              rngs)

    def test_actor_critic_independently_initialized(self):
        net = self.mlp_net
        
        # Test actor and critic were independently initialized
        for actor_layer, critic_layer in zip(net.actor.layers, net.critic.layers):
            self.assertFalse(jp.allclose(actor_layer.kernel.raw_value,
                                         critic_layer.kernel.raw_value))

    def test_actor_layers_independently_initialized(self):
        SEED = 123
        self.obs_size = 64
        self.action_size = 32
        self.hidden_sizes = [64, 64]
        rngs = nnx.Rngs(SEED)
        net = modules.MLPActorCritic(self.obs_size,
                                     self.action_size,
                                     self.hidden_sizes,
                                     self.hidden_sizes,
                                     rngs)
        
        # Test actor and critic were independently initialized
        first_actor_layer = net.actor.layers[0]
        for actor_layer in net.actor.layers[1:]:
            self.assertFalse(jp.allclose(first_actor_layer.kernel.raw_value,
                                         actor_layer.kernel.raw_value))

    def test_init_state(self):
        net = self.mlp_net

        # Init
        first_state = net.initialize_state(batch_size=17)
        self.assertDictContainsSubset({"actor": (), "critic": ()}, first_state)


    def test_simple_input(self):
        net = self.mlp_net
        
        key = jax.random.key(seed=18)
        simple_obs = jax.random.normal(key, (1, self.obs_size))

        first_state = net.initialize_state(1)
        next_state, output = net(first_state, simple_obs)
        self.assertIsInstance(output, types.PPONetworkOutput)

    def test_simple_input_jit(self):
        @nnx.jit
        def init_state(net, batch_size: int):
            return net.initialize_state(batch_size)
        
        @nnx.jit
        def call_net(net, state, x):
            return net(state, x)
        
        net = self.mlp_net
        key = jax.random.key(seed=19)
        simple_obs = jax.random.normal(key, (1, self.obs_size))

        first_state = init_state(net, 1)
        next_state, output = call_net(net, first_state, simple_obs)
        self.assertIsInstance(output, types.PPONetworkOutput)

        next_state, output = call_net(net, next_state, simple_obs)
        self.assertIsInstance(output, types.PPONetworkOutput)
        self.assertDictContainsSubset({"actor": (), "critic": ()}, next_state)

    def test_simple_input_batched(self):
        net = self.mlp_net
        n_envs = 256
        key = jax.random.key(seed=21)
        simple_obs = jax.random.normal(key, (n_envs, self.obs_size))

        first_state = net.initialize_state(n_envs)
        next_state, first_output = net(first_state, simple_obs)
        self.assertIsInstance(first_output, types.PPONetworkOutput)
        self.assertEqual(first_output.actions.shape, (n_envs, self.action_size))
        self.assertEqual(first_output.loglikelihoods.shape, (n_envs,))
        self.assertEqual(first_output.value_estimates.shape, (n_envs, 1))
        self.assertLess(jp.min(first_output.value_estimates), -0.2)
        self.assertGreater(jp.max(first_output.value_estimates), 0.2)

        next_state, second_output = net(next_state, simple_obs)
        self.assertIsInstance(second_output, types.PPONetworkOutput)
        self.assertDictContainsSubset({"actor": (), "critic": ()}, next_state)
        #self.assertFalse(jp.allclose(next_state["action_sampler"], first_state["action_sampler"]))
        self.assertFalse(jp.allclose(first_output.actions, second_output.actions),
                         "Stochasticity in actions.")
        self.assertTrue(jp.allclose(first_output.value_estimates, second_output.value_estimates),
                        "No stochasticity in critic.")

    def test_action_sampler_train_and_eval_mode(self):
        net = self.mlp_net
        self.assertFalse(net.action_sampler.deterministic)
        net.eval()
        self.assertTrue(net.action_sampler.deterministic)
        net.train()
        self.assertFalse(net.action_sampler.deterministic)

    def test_normalize_obs(self):
        SEED = 42
        OBS_SIZE = 24
        ACTION_SIZE = 5
        BATCH_SIZE = 32
        N_STEPS = 10
        nets = modules.MLPActorCritic(OBS_SIZE, ACTION_SIZE,
                                      actor_hidden_sizes=[64, 64],
                                      critic_hidden_sizes=[64, 64],
                                      rngs = nnx.Rngs(SEED, action_sampling=SEED),
                                      normalize_obs=True)
        key = jax.random.key(SEED)
        mean_key, var_key = jax.random.split(key)
        data = jax.random.normal(mean_key, (OBS_SIZE,)) + jax.random.normal(var_key, (N_STEPS+1, BATCH_SIZE, OBS_SIZE))
        state = nets.initialize_state(BATCH_SIZE)
        for i in range(N_STEPS):
            state, _ = nets(state, data[i])
            self.assertEqual(nets.preprocessor.counter.value, (i+1) * BATCH_SIZE)
            self.assertEqual(nets.preprocessor.mean.value.shape, (OBS_SIZE,))
            self.assertEqual(nets.preprocessor.M2.value.shape, (OBS_SIZE,))

            true_mean = jp.mean(data[:i+1], axis=(0, 1))
            true_std  = jp.std(data[:i+1], axis=(0, 1))
            est_mean = nets.preprocessor.mean
            est_var = nets.preprocessor.M2 / nets.preprocessor.counter
            est_std = jp.sqrt(est_var)
            self.assertLess(jp.max(jp.abs(est_mean - true_mean)), 1e-6)
            self.assertLess(jp.max(jp.abs(est_std - true_std)), 1e-6)

        # Check that the normalizer isn't updated in eval mode
        nets.eval()
        state, _ = nets(state, data[i])
        self.assertEqual(nets.preprocessor.counter.value, N_STEPS * BATCH_SIZE)
        est_mean = nets.preprocessor.mean
        est_var = nets.preprocessor.M2 / nets.preprocessor.counter
        est_std = jp.sqrt(est_var)
        self.assertLess(jp.max(jp.abs(est_mean - true_mean)), 1e-6)
        self.assertLess(jp.max(jp.abs(est_std - true_std)), 1e-6)
        
if __name__ == '__main__':
    absltest.main()