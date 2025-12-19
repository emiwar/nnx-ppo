import functools
from typing import Tuple, Any, Dict

from absl.testing import absltest
import jax
import jax.numpy as jp
from flax import nnx
import mujoco_playground

from nnx_ppo.networks import modules, types
from nnx_ppo.algorithms.rollout import unroll_env
from nnx_ppo.test_dummies import parrot_env
from nnx_ppo.wrappers import episode_wrapper

class EpisodeWrapperTest(absltest.TestCase):

    def setUp(self):
        SEED = 17

        base_env = parrot_env.ParrotEnv()
        self.env = episode_wrapper.EpisodeWrapper(base_env, max_len=10)
        self.nets = modules.MLPActorCritic(self.env.observation_size, self.env.action_size,
                                           actor_hidden_sizes=[16, 16],
                                           critic_hidden_sizes=[16, 16],
                                           rngs = nnx.Rngs(SEED, action_sampling=SEED))
        
    def test_single_env_rollout(self):
        N_STEPS = 24
        key = jax.random.key(seed=18)
        net_key, env_key, reset_key = jax.random.split(key, 3)
        net_state = self.nets.initialize_state(1)
        env_state = jax.vmap(self.env.reset)(jax.random.split(env_key, 1))
        self.assertTrue("step_counter" in env_state.info)

        next_net_state, next_env_state, rollout_data = unroll_env(
            self.env, env_state, self.nets, net_state, N_STEPS, reset_key)
        self.assertTrue("step_counter" in next_env_state.info)

    def test_vmap_env_rollout(self):
        N_ENVS = 256
        N_STEPS = 24
        key = jax.random.key(seed=18)
        net_key, env_key, reset_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, N_ENVS)
        env_states = jax.vmap(self.env.reset)(env_keys)
        net_states = self.nets.initialize_state(N_ENVS)

        next_net_state, next_env_state, rollout_data = unroll_env(
            self.env, env_states, self.nets, net_states, N_STEPS, reset_key)
        self.assertTrue(jp.all(next_env_state.info["step_counter"] <= self.env.max_len))
        self.assertFalse(jp.all(next_env_state.info["step_counter"] == 0))
