import functools
from typing import Tuple, Any, Dict

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jp
from flax import nnx
import mujoco_playground

from nnx_ppo.networks import modules, types
from nnx_ppo.algorithms.rollout import unroll_env

class RolloutTest(absltest.TestCase):

    def setUp(self):
        SEED = 17

        self.env = mujoco_playground.registry.load("CartpoleBalance")
        self.nets = modules.MLPActorCritic(self.env.observation_size, self.env.action_size,
                                           actor_hidden_sizes=[16, 16],
                                           critic_hidden_sizes=[16, 16],
                                           rngs = nnx.Rngs(SEED, action_sampling=SEED))
        
    def test_single_env_rollout(self):
        N_STEPS = 24
        key = jax.random.key(seed=18)
        net_key, env_key, reset_key = jax.random.split(key, 3)
        net_state = self.nets.initialize_state(net_key)
        env_state = self.env.reset(env_key)

        next_net_state, next_env_state, rollout_data = unroll_env(
            self.env, env_state, self.nets, net_state, N_STEPS, reset_key)
        self.assertEquals(rollout_data.done.shape, (N_STEPS,))
        self.assertEquals(rollout_data.rewards.shape, (N_STEPS,))
        self.assertEquals(rollout_data.obs.shape, (N_STEPS, self.env.observation_size))
        self.assertEquals(rollout_data.next_obs.shape, (N_STEPS, self.env.observation_size))
        self.assertEquals(rollout_data.network_output.actions.shape, (N_STEPS, self.env.action_size))
        self.assertEquals(rollout_data.network_output.loglikelihoods.shape, (N_STEPS,))

    def test_single_env_rollout_jit(self):
        N_STEPS = 24
        key = jax.random.key(seed=18)
        
        net_key, env_key, reset_key = jax.random.split(key, 3)
        net_state = self.nets.initialize_state(net_key)
        env_state = self.env.reset(env_key)

        unroll_env_jit = nnx.jit(functools.partial(unroll_env, self.env), static_argnames=("unroll_length"))
        next_net_state, next_env_state, rollout_data = unroll_env_jit(
            env_state, self.nets, net_state, N_STEPS, reset_key)
        self.assertEquals(rollout_data.done.shape, (N_STEPS,))
        self.assertEquals(rollout_data.rewards.shape, (N_STEPS,))
        self.assertEquals(rollout_data.obs.shape, (N_STEPS, self.env.observation_size))
        self.assertEquals(rollout_data.next_obs.shape, (N_STEPS, self.env.observation_size))
        self.assertEquals(rollout_data.network_output.actions.shape, (N_STEPS, self.env.action_size))
        self.assertEquals(rollout_data.network_output.loglikelihoods.shape, (N_STEPS,))

        #Do another rollout just to be sure there're no stray tracers
        unroll_env_jit(next_env_state, self.nets, next_net_state, N_STEPS, reset_key)

    def test_vmap_env_rollout(self):
        N_ENVS = 256
        N_STEPS = 24
        key = jax.random.key(seed=18)
        net_key, env_key, reset_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, N_ENVS)
        env_states = jax.vmap(self.env.reset)(env_keys)
        net_init_keys = jax.random.split(net_key, N_ENVS)
        net_states = nnx.vmap(self.nets.initialize_state)(net_init_keys)
        reset_keys = jax.random.split(reset_key, N_ENVS)

        unroll_vmap = nnx.vmap(unroll_env, in_axes=(None, 0, None, 0, None, 0), out_axes=(0, 0, 0))

        next_net_state, next_env_state, rollout_data = unroll_vmap(
            self.env, env_states, self.nets, net_states, N_STEPS, reset_keys)
        self.assertEquals(rollout_data.done.shape, (N_ENVS, N_STEPS,))
        self.assertEquals(rollout_data.rewards.shape, (N_ENVS, N_STEPS,))
        self.assertEquals(rollout_data.obs.shape, (N_ENVS, N_STEPS, self.env.observation_size))
        self.assertEquals(rollout_data.next_obs.shape, (N_ENVS, N_STEPS, self.env.observation_size))
        self.assertEquals(rollout_data.network_output.actions.shape, (N_ENVS, N_STEPS, self.env.action_size))
        self.assertEquals(rollout_data.network_output.loglikelihoods.shape, (N_ENVS, N_STEPS,))
        jax.tree.map(lambda a, b: self.assertEquals(a.shape, b.shape), net_states, next_net_state)
        jax.tree.map(lambda a, b: self.assertEquals(a.shape, b.shape), env_states, next_env_state)
        unroll_vmap(self.env, next_env_state, self.nets, next_net_state, N_STEPS, reset_keys)

    def test_dummy_env_rollout(self):
        N_STEPS = 24
        dummy_env = DummyCounterEnv()
        dummy_nets = DummyCounterNet()
        key = jax.random.key(seed=18)
        net_key, env_key, reset_key = jax.random.split(key, 3)
        net_state = dummy_nets.initialize_state(net_key)
        env_state = dummy_env.reset(env_key)

        next_net_state, next_env_state, rollout_data = unroll_env(
            dummy_env, env_state, dummy_nets, net_state, N_STEPS, reset_key)
        self.assertEquals(rollout_data.done.shape, (N_STEPS,))
        self.assertEquals(rollout_data.rewards.shape, (N_STEPS,))
        self.assertEquals(rollout_data.network_output.loglikelihoods.shape, (N_STEPS,))

        self.assertGreaterEqual(jp.sum(rollout_data.done), 2)
        self.assertLess(jp.sum(rollout_data.done), 10)
        self.assertEqual(jp.sum(rollout_data.rewards), N_STEPS)

    def test_dummy_env_rollout_jit(self):
        N_STEPS = 24
        dummy_env = DummyCounterEnv()
        dummy_nets = DummyCounterNet()
        key = jax.random.key(seed=18)
        net_key, env_key, reset_key = jax.random.split(key, 3)
        net_state = dummy_nets.initialize_state(net_key)
        env_state = dummy_env.reset(env_key)

        unroll_env_jit = nnx.jit(unroll_env, static_argnames=("env", "unroll_length"))
        next_net_state, next_env_state, rollout_data = unroll_env_jit(
            dummy_env, env_state, dummy_nets, net_state, N_STEPS, reset_key)
        self.assertEquals(rollout_data.done.shape, (N_STEPS,))
        self.assertEquals(rollout_data.rewards.shape, (N_STEPS,))
        self.assertEquals(rollout_data.network_output.loglikelihoods.shape, (N_STEPS,))

        self.assertGreaterEqual(jp.sum(rollout_data.done), 2)
        self.assertLess(jp.sum(rollout_data.done), 10)
        self.assertEqual(jp.sum(rollout_data.rewards), N_STEPS)

    def test_dummy_env_rollout_vmap(self):
        N_ENVS = 256
        N_STEPS = 24
        dummy_env = DummyCounterEnv()
        dummy_nets = DummyCounterNet()

        key = jax.random.key(seed=15)
        net_key, env_key, reset_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, N_ENVS)
        env_states = jax.vmap(dummy_env.reset)(env_keys)
        net_init_keys = jax.random.split(net_key, N_ENVS)
        net_states = nnx.vmap(dummy_nets.initialize_state)(net_init_keys)
        reset_keys = jax.random.split(reset_key, N_ENVS)

        unroll_vmap = nnx.vmap(unroll_env, in_axes=(None, 0, None, 0, None, 0), out_axes=(0, 0, 0))

        next_net_state, next_env_state, rollout_data = unroll_vmap(
            dummy_env, env_states, dummy_nets, net_states, N_STEPS, reset_keys)
        self.assertEquals(rollout_data.done.shape, (N_ENVS, N_STEPS))
        self.assertEquals(rollout_data.rewards.shape, (N_ENVS, N_STEPS))
        self.assertEquals(rollout_data.network_output.loglikelihoods.shape, (N_ENVS, N_STEPS))

        self.assertGreaterEqual(jp.sum(rollout_data.done), 2*N_ENVS)
        self.assertLess(jp.sum(rollout_data.done), 10*N_ENVS)
        self.assertEqual(jp.sum(rollout_data.rewards), N_STEPS*N_ENVS)
        jax.tree.map(lambda a, b: self.assertEquals(a.shape, b.shape), net_states, next_net_state)
        jax.tree.map(lambda a, b: self.assertEquals(a.shape, b.shape), env_states, next_env_state)

class DummyCounterEnv:
    '''Dummy environment that gives reward 1.0 if the action is the number
       of steps since the last reset, and 0.0 otherwise. Observation is always
       [0.0].'''

    def reset(self, rng):
        return mujoco_playground.State(
            data = {"current_step": jp.array(0),
                    "reset_step": jax.random.randint(rng, (), 3, 10)},
            obs = jp.zeros(1),
            info = {},
            reward = jp.array(1.0),
            done = jp.array(0.0),
            metrics = {}
        )
    
    def step(self, state: mujoco_playground.State, action: jax.Array):
        data = {"current_step": state.data["current_step"] + 1,
                "reset_step": state.data["reset_step"]}
        done = jp.astype(data["current_step"] >= data["reset_step"], float)
        return mujoco_playground.State(
            data = data,
            obs = jp.zeros(1),
            info={},
            reward=jp.where(action==data["current_step"], 1.0, 0.0),
            done=done,
            metrics=state.metrics
        )
    
class DummyCounterNet(types.AbstractPPOActorCritic, nnx.Module):
    '''Dummy stateful network that always outputs the number of steps since its
       last reset, independent of its input.'''
    
    def __call__(self, state, obs) -> Tuple[Any, types.PPONetworkOutput]:
        old_counter = state["counter_state"]["counter"]
        new_state = {"counter_state": {"counter": old_counter + 1}}
        return new_state, types.PPONetworkOutput(
            actions=old_counter + 1,
            loglikelihoods=jp.array(1.0),
            regularization_loss=jp.array(0.0),
            value_estimates=jp.array(1.0),
            metrics={}
        )
    
    def initialize_state(self, rng: jax.Array) -> Dict:
        return {"counter_state": {"counter": 0}}
    
    def reset_state(self, old_state) -> Dict:
        return {"counter_state": {"counter": 0}}
