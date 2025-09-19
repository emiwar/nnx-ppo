from absl.testing import absltest, parameterized
import jax
import jax.numpy as jp
from flax import nnx

from networks import modules, types

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

    def test_layers_independently_initialized(self):
        net = self.mlp_net
        
        # Test actor and critic were independently initialized
        for actor_layer, critic_layer in zip(net.actor.layers, net.critic.layers):
            self.assertFalse(jp.allclose(actor_layer.kernel, critic_layer.kernel))
            #self.assertFalse(jp.allclose(actor_layer.bias, critic_layer.bias))


    def test_init_state(self):

        net = self.mlp_net
        
        key = jax.random.key(seed=17)

        # Init
        first_state = net.initialize_state(key)
        self.assertDictContainsSubset({"actor": (), "critic": ()}, first_state)

        # Reset
        second_state = net.reset_state(first_state)
        self.assertEqual(first_state, second_state)

    def test_simple_input(self):
        net = self.mlp_net
        
        key = jax.random.key(seed=18)
        obs_key, net_init_key = jax.random.split(key)
        simple_obs = jax.random.normal(obs_key, (self.obs_size,))

        first_state = net.initialize_state(net_init_key)
        next_state, output = net(first_state, simple_obs)
        self.assertIsInstance(output, types.PPONetworkOutput)

        # TODO: Check output more!

    def test_simple_input_jit(self):
        @nnx.jit
        def init_state(net, key):
            return net.initialize_state(key)
        
        @nnx.jit
        def call_net(net, state, x):
            return net(state, x)
        
        @nnx.jit
        def reset_state(net, state):
            return net.reset_state(state)
        
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
        self.assertNotEqual(next_state["action_sampler"], first_state["action_sampler"])

        next_state = reset_state(net, next_state)
        self.assertDictContainsSubset({"actor": (), "critic": ()}, next_state)
        self.assertNotEqual(next_state["action_sampler"], first_state["action_sampler"])

    def test_init_state_vmap(self):
        @nnx.vmap(in_axes=(None, 0))
        def init_state(net, key):
            return net.initialize_state(key)
        
        @nnx.vmap(in_axes=(None, 0))
        def reset_state(net, states):
            return net.reset_state(states)

        n_envs = 256
        net = self.mlp_net
        key = jax.random.key(seed=19)

        # Init
        init_keys = jax.random.split(key, n_envs)
        first_state = init_state(net, init_keys)
        self.assertDictContainsSubset({"actor": (), "critic": ()}, first_state)
        self.assertEquals(first_state["action_sampler"].shape[0], n_envs)

        # Reset
        second_state = reset_state(net, first_state)
        self.assertDictEqual(first_state, second_state)

    def test_simple_input_vmap(self):
        @nnx.vmap(in_axes=(None, 0))
        def init_state(net, key):
            return net.initialize_state(key)
        
        @nnx.vmap(in_axes=(None, 0, 0))
        def call_net(net, state, x):
            return net(state, x)
        
        @nnx.vmap(in_axes=(None, 0))
        def reset_state(net, state):
            return net.reset_state(state)
        
        net = self.mlp_net
        n_envs = 256
        key = jax.random.key(seed=21)
        obs_key, net_init_key = jax.random.split(key)
        simple_obs = jax.random.normal(obs_key, (n_envs, self.obs_size))

        init_keys = jax.random.split(net_init_key, n_envs)
        first_state = init_state(net, init_keys)
        next_state, first_output = call_net(net, first_state, simple_obs)
        self.assertIsInstance(first_output, types.PPONetworkOutput)
        self.assertEquals(first_output.actions.shape, (n_envs, self.action_size))
        self.assertEquals(first_output.loglikelihoods.shape, (n_envs,))
        self.assertEquals(first_output.value_estimates.shape, (n_envs, 1))

        next_state, second_output = call_net(net, next_state, simple_obs)
        self.assertIsInstance(second_output, types.PPONetworkOutput)
        self.assertDictContainsSubset({"actor": (), "critic": ()}, next_state)
        self.assertFalse(jp.allclose(next_state["action_sampler"], first_state["action_sampler"]))
        self.assertFalse(jp.allclose(first_output.actions, second_output.actions),
                         "Stochasticity in actions.")
        self.assertTrue(jp.allclose(first_output.value_estimates, second_output.value_estimates),
                        "No stochasticity in critic.")

        next_state = reset_state(net, next_state)
        self.assertDictContainsSubset({"actor": (), "critic": ()}, next_state)
        self.assertFalse(jp.allclose(next_state["action_sampler"], first_state["action_sampler"]))

if __name__ == '__main__':
    absltest.main()