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
        obs_key, net_init_key = jax.random.split(key)
        simple_obs = jax.random.normal(obs_key, (self.obs_size,))

        first_state = net.initialize_state(net_init_key)
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
        key = jax.random.key(seed=18)
        obs_key, net_init_key = jax.random.split(key)
        simple_obs = jax.random.normal(obs_key, (self.obs_size,))

        first_state = init_state(net, net_init_key)
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
        self.assertEquals(first_output.actions.shape, (n_envs, self.action_size))
        self.assertEquals(first_output.loglikelihoods.shape, (n_envs,))
        self.assertEquals(first_output.value_estimates.shape, (n_envs, 1))
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

if __name__ == '__main__':
    absltest.main()