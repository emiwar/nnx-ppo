import functools
from typing import Tuple, Any, Dict

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jp
from flax import nnx
import mujoco_playground

from nnx_ppo.networks import modules, types
from nnx_ppo.algorithms.rollout import unroll_env
from nnx_ppo.test_dummies import dummy_counter
from nnx_ppo.test_dummies import stateful_nets, parrot_env, move_to_center_env
class RolloutTest(absltest.TestCase):

    def setUp(self):
        SEED = 17

        self.env = mujoco_playground.registry.load("CartpoleSwingup")
        self.nets = modules.MLPActorCritic(self.env.observation_size, self.env.action_size,
                                           actor_hidden_sizes=[16, 16],
                                           critic_hidden_sizes=[16, 16],
                                           rngs = nnx.Rngs(SEED, action_sampling=SEED))
        
    def test_single_env_rollout(self):
        N_STEPS = 24
        key = jax.random.key(seed=18)
        env_key, reset_key = jax.random.split(key)
        net_state = self.nets.initialize_state(batch_size=1)
        env_state = jax.vmap(self.env.reset)(jax.random.split(env_key, 1))

        next_net_state, next_env_state, rollout_data = unroll_env(
            self.env, env_state, self.nets, net_state, N_STEPS, reset_key)
        self.assertEquals(rollout_data.done.shape, (N_STEPS,1))
        self.assertEquals(rollout_data.rewards.shape, (N_STEPS,1))
        self.assertEquals(rollout_data.obs.shape, (N_STEPS, 1, self.env.observation_size))
        self.assertEquals(rollout_data.next_obs.shape, (N_STEPS, 1, self.env.observation_size))
        self.assertEquals(rollout_data.network_output.actions.shape, (N_STEPS, 1, self.env.action_size))
        self.assertEquals(rollout_data.network_output.loglikelihoods.shape, (N_STEPS, 1))

    def test_single_env_rollout_jit(self):
        N_STEPS = 24
        key = jax.random.key(seed=18)
        env_key, reset_key = jax.random.split(key)
        net_state = self.nets.initialize_state(batch_size=1)
        env_state = jax.vmap(self.env.reset)(jax.random.split(env_key, 1))

        unroll_env_jit = nnx.jit(functools.partial(unroll_env, self.env), static_argnames=("unroll_length"))
        next_net_state, next_env_state, rollout_data = unroll_env_jit(
            env_state, self.nets, net_state, N_STEPS, reset_key)
        self.assertEquals(rollout_data.done.shape, (N_STEPS,1))
        self.assertEquals(rollout_data.rewards.shape, (N_STEPS,1))
        self.assertEquals(rollout_data.obs.shape, (N_STEPS, 1, self.env.observation_size))
        self.assertEquals(rollout_data.next_obs.shape, (N_STEPS, 1, self.env.observation_size))
        self.assertEquals(rollout_data.network_output.actions.shape, (N_STEPS, 1, self.env.action_size))
        self.assertEquals(rollout_data.network_output.loglikelihoods.shape, (N_STEPS, 1))

        #Do another rollout just to be sure there're no stray tracers
        unroll_env_jit(next_env_state, self.nets, next_net_state, N_STEPS, reset_key)

    def test_batch_env_rollout(self):
        N_ENVS = 256
        N_STEPS = 24
        key = jax.random.key(seed=18)
        env_key, reset_key = jax.random.split(key)
        net_states = self.nets.initialize_state(batch_size=N_ENVS)
        env_states = jax.vmap(self.env.reset)(jax.random.split(env_key, N_ENVS))

        next_net_state, next_env_state, rollout_data = unroll_env(
            self.env, env_states, self.nets, net_states, N_STEPS, reset_key)
        self.assertEquals(rollout_data.done.shape, (N_STEPS, N_ENVS))
        self.assertEquals(rollout_data.rewards.shape, (N_STEPS, N_ENVS))
        self.assertEquals(rollout_data.obs.shape, (N_STEPS, N_ENVS, self.env.observation_size))
        self.assertEquals(rollout_data.next_obs.shape, (N_STEPS, N_ENVS, self.env.observation_size))
        self.assertEquals(rollout_data.network_output.actions.shape, (N_STEPS, N_ENVS, self.env.action_size))
        self.assertEquals(rollout_data.network_output.loglikelihoods.shape, (N_STEPS, N_ENVS))
        jax.tree.map(lambda a, b: self.assertEquals(a.shape, b.shape), net_states, next_net_state)
        jax.tree.map(lambda a, b: self.assertEquals(a.shape, b.shape), env_states, next_env_state)
        unroll_env(self.env, next_env_state, self.nets, next_net_state, N_STEPS, reset_key)

    def test_dummy_env_rollout(self):
        N_ENVS = 1
        N_STEPS = 24
        dummy_env = dummy_counter.DummyCounterEnv()
        dummy_nets = dummy_counter.DummyCounterNet()
        key = jax.random.key(seed=18)
        net_key, env_key, reset_key = jax.random.split(key, 3)
        net_state = dummy_nets.initialize_state(batch_size=N_ENVS)
        env_states = jax.vmap(dummy_env.reset)(jax.random.split(env_key, N_ENVS))

        next_net_state, next_env_state, rollout_data = unroll_env(
            dummy_env, env_states, dummy_nets, net_state, N_STEPS, reset_key)
        self.assertEquals(rollout_data.done.shape, (N_STEPS, 1))
        self.assertEquals(rollout_data.rewards.shape, (N_STEPS, 1))
        self.assertEquals(rollout_data.network_output.loglikelihoods.shape, (N_STEPS, 1))

        self.assertGreaterEqual(jp.sum(rollout_data.done), 2)
        self.assertLess(jp.sum(rollout_data.done), 10)
        self.assertEqual(jp.sum(rollout_data.rewards), N_STEPS)

    def test_dummy_env_rollout_jit(self):
        N_ENVS = 1
        N_STEPS = 24
        dummy_env = dummy_counter.DummyCounterEnv()
        dummy_nets = dummy_counter.DummyCounterNet()
        key = jax.random.key(seed=18)
        net_key, env_key, reset_key = jax.random.split(key, 3)
        net_state = dummy_nets.initialize_state(batch_size=N_ENVS)
        env_state = jax.vmap(dummy_env.reset)(jax.random.split(env_key, N_ENVS))

        unroll_env_jit = nnx.jit(unroll_env, static_argnames=("env", "unroll_length"))
        next_net_state, next_env_state, rollout_data = unroll_env_jit(
            dummy_env, env_state, dummy_nets, net_state, N_STEPS, reset_key)
        self.assertEquals(rollout_data.done.shape, (N_STEPS, 1))
        self.assertEquals(rollout_data.rewards.shape, (N_STEPS, 1))
        self.assertEquals(rollout_data.network_output.loglikelihoods.shape, (N_STEPS, 1))

        self.assertGreaterEqual(jp.sum(rollout_data.done), 2)
        self.assertLess(jp.sum(rollout_data.done), 10)
        self.assertEqual(jp.sum(rollout_data.rewards), N_STEPS)

    def test_dummy_env_rollout_batch(self):
        N_ENVS = 256
        N_STEPS = 24
        dummy_env = dummy_counter.DummyCounterEnv()
        dummy_nets = dummy_counter.DummyCounterNet()
        key = jax.random.key(seed=19)
        net_key, env_key, reset_key = jax.random.split(key, 3)
        net_states = dummy_nets.initialize_state(batch_size=N_ENVS)
        env_states = jax.vmap(dummy_env.reset)(jax.random.split(env_key, N_ENVS))

        unroll_env_jit = nnx.jit(unroll_env, static_argnames=("env", "unroll_length"))
        next_net_state, next_env_state, rollout_data = unroll_env_jit(
            dummy_env, env_states, dummy_nets, net_states, N_STEPS, reset_key)
        self.assertEquals(rollout_data.done.shape, (N_STEPS, N_ENVS))
        self.assertEquals(rollout_data.rewards.shape, (N_STEPS, N_ENVS))
        self.assertEquals(rollout_data.network_output.loglikelihoods.shape, (N_STEPS, N_ENVS))

        self.assertGreaterEqual(jp.sum(rollout_data.done), 2*N_ENVS)
        self.assertLess(jp.sum(rollout_data.done), 10*N_ENVS)
        self.assertEqual(jp.sum(rollout_data.rewards), N_STEPS*N_ENVS)
        jax.tree.map(lambda a, b: self.assertEquals(a.shape, b.shape), net_states, next_net_state)
        jax.tree.map(lambda a, b: self.assertEquals(a.shape, b.shape), env_states, next_env_state)

    def test_basic_stateful_net(self):
        N_ENVS = 1
        N_STEPS = 24
        net = stateful_nets.RepeatAndCountNet()
        env = parrot_env.ParrotEnv()
        key = jax.random.key(seed=18)
        net_key, env_key, reset_key = jax.random.split(key, 3)
        net_state = net.initialize_state(N_ENVS)
        env_state = jax.vmap(env.reset)(jax.random.split(env_key, N_ENVS))

        next_net_state, next_env_state, rollout_data = unroll_env(
            env, env_state, net, net_state, N_STEPS, reset_key)
        self.assertEquals(net.n_calls.value, N_STEPS * N_ENVS)

    def test_stateful_net_batch(self):
        N_ENVS = 256
        N_STEPS = 24
        net = stateful_nets.RepeatAndCountNet()
        env = parrot_env.ParrotEnv()
        key = jax.random.key(seed=18)
        net_key, env_key, reset_key = jax.random.split(key, 3)
        net_state = net.initialize_state(N_ENVS)
        env_state = jax.vmap(env.reset)(jax.random.split(env_key, N_ENVS))

        next_net_state, next_env_state, rollout_data = unroll_env(
            env, env_state, net, net_state, N_STEPS, reset_key)
        self.assertEquals(net.n_calls.value, N_STEPS * N_ENVS)

    def test_rng_stream_resets(self):
        SEED = 23
        N_ENVS = 64
        N_STEPS = 30
        key = jax.random.key(SEED)
        env = move_to_center_env.MoveToCenterEnv(reward_falloff=1.0, border_radius=10.0)
        net = modules.MLPActorCritic(env.observation_size, env.action_size,
                                      actor_hidden_sizes=[128, 128],
                                      critic_hidden_sizes=[128, 128],
                                      rngs = nnx.Rngs(SEED, action_sampling=SEED))
        net_key, env_key, reset_key = jax.random.split(key, 3)
        net_state = net.initialize_state(N_ENVS)
        env_state = jax.vmap(env.reset)(jax.random.split(env_key, N_ENVS))
        pre_rollout_module_state = nnx.state(net)

        #Same env & net state, but the RNG stream in the network should advance
        next_net_state, next_env_state, rollout_data1 = unroll_env(
            env, env_state, net, net_state, N_STEPS, reset_key)
        next_net_state, next_env_state, rollout_data2 = unroll_env(
            env, env_state, net, net_state, N_STEPS, reset_key)
        self.assertGreater(jp.mean(jp.abs(rollout_data1.network_output.actions - rollout_data2.network_output.actions)), 0.2)
        self.assertGreater(jp.mean(jp.abs(rollout_data1.network_output.loglikelihoods - rollout_data2.network_output.loglikelihoods)), 0.2)

        #Resetting the state of the RNG stream
        nnx.update(net, pre_rollout_module_state)
        next_net_state, next_env_state, rollout_data3 = unroll_env(
            env, env_state, net, net_state, N_STEPS, reset_key)
        self.assertLess(jp.mean(jp.abs(rollout_data1.network_output.actions - rollout_data3.network_output.actions)), 1e-6)
        self.assertLess(jp.mean(jp.abs(rollout_data1.network_output.loglikelihoods - rollout_data3.network_output.loglikelihoods)), 1e-6)