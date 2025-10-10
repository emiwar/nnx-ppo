import functools
from typing import Tuple, Any, Dict

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jp
import numpy as np
from jax.experimental import checkify
from flax import nnx
import mujoco_playground

from nnx_ppo.networks import modules, types
from nnx_ppo.algorithms import rollout, ppo

class PPOTest(absltest.TestCase):

    def setUp(self):
        SEED = 17
        self.env = mujoco_playground.registry.load("CartpoleBalance")
        self.nets = modules.MLPActorCritic(self.env.observation_size, self.env.action_size,
                                           actor_hidden_sizes=[16, 16],
                                           critic_hidden_sizes=[16, 16],
                                           rngs = nnx.Rngs(SEED, action_sampling=SEED))

    def test_ppo_step(self):
        SEED = 18
        config = ppo.default_config()
        training_state = ppo.new_training_state(self.env, self.nets, config.n_envs, SEED)
        self.assertEquals(training_state.steps_taken, 0)
        training_state, _ = ppo.ppo_step(self.env, training_state, config.n_envs, config.rollout_length,
                                         config.gae_lambda, config.discounting_factor, config.clip_range,
                                         config.normalize_advantages)
        self.assertEquals(training_state.steps_taken, config.n_envs*config.rollout_length)

    def test_ppo_step_twice(self):
        SEED = 18
        config = ppo.default_config()
        training_state = ppo.new_training_state(self.env, self.nets, config.n_envs, SEED)
        self.assertEquals(training_state.steps_taken, 0)
        training_state, _ = ppo.ppo_step(self.env, training_state, config.n_envs, config.rollout_length,
                                         config.gae_lambda, config.discounting_factor, config.clip_range,
                                         config.normalize_advantages)
        self.assertEquals(training_state.steps_taken, config.n_envs*config.rollout_length)
        training_state, _ = ppo.ppo_step(self.env, training_state, config.n_envs, config.rollout_length,
                                         config.gae_lambda, config.discounting_factor, config.clip_range,
                                         config.normalize_advantages)
        self.assertEquals(training_state.steps_taken, config.n_envs*config.rollout_length*2)

    def test_ppo_step_jit(self):
        SEED = 18
        config = ppo.default_config()
        training_state = nnx.jit(ppo.new_training_state, static_argnums=(0, 2, 3))(self.env, self.nets, config.n_envs, SEED)
        self.assertEquals(training_state.steps_taken, 0)
        #ppo_step_fcn = functools.partial(ppo.ppo_step, self.env)
        ppo_step_fcn = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3, 7))
        #ppo_step_fcn = checkify.checkify(ppo_step_fcn)
        training_state, _ = ppo_step_fcn(self.env, training_state, config.n_envs, config.rollout_length,
                                         config.gae_lambda, config.discounting_factor, config.clip_range,
                                         config.normalize_advantages)
        #err.throw()
        self.assertEquals(training_state.steps_taken, config.n_envs*config.rollout_length)
        training_state, _ = ppo_step_fcn(self.env, training_state, config.n_envs, config.rollout_length,
                                         config.gae_lambda, config.discounting_factor, config.clip_range,
                                         config.normalize_advantages)
        #err.throw()
        self.assertEquals(training_state.steps_taken, config.n_envs*config.rollout_length*2)


    def test_rollout_only(self):
        SEED = 18
        config = ppo.default_config()
        training_state = nnx.jit(ppo.new_training_state, static_argnums=(0, 2))(self.env, self.nets, config.n_envs, SEED)
        unroll_jit = nnx.jit(rollout.unroll_env, static_argnums=(0, 4))
        unroll_jit(self.env, training_state.env_states, training_state.networks,
                   training_state.network_states, config.rollout_length,
                   training_state.rng_key)

    def test_ppo_full_loop(self):
        SEED = 21
        ppo.train_ppo(self.env, self.nets, ppo.default_config(), SEED)

    def test_gae(self):
        SEED = 23
        N_ENVS = 512
        T_STEPS = 100
        gamma = 0.8
        lambda_ = 0.95

        np.random.seed(SEED)
        rewards = np.random.normal(size=(T_STEPS, N_ENVS))
        values = np.random.normal(size=(T_STEPS+1, N_ENVS))
        done = np.random.choice([True, False], size=(T_STEPS, N_ENVS), p=[0.01, 0.99])
        truncation = np.random.choice([True, False], size=(T_STEPS, N_ENVS))
        truncation = np.logical_and(done, truncation)
    
        advantages = np.full((T_STEPS, N_ENVS), np.nan)
        for t in reversed(range(T_STEPS)):
            next_values = values[t+1, :].copy()
            next_values[done[t, :]] = 0.0
            next_values[truncation[t, :]] = values[t, truncation[t, :]]
            advantages[t, :] = rewards[t, :] + gamma * next_values - values[t, :]
            if t < T_STEPS-1:
                advantages[t, :] += gamma * lambda_ * advantages[t+1, :] * (1 - done[t, :])

        ppo_advantages = ppo.gae(rewards=jp.array(rewards),
                                 values=jp.array(values),
                                 done=jp.array(done, dtype=bool),
                                 truncation=jp.array(truncation, dtype=bool),
                                 lambda_=lambda_,
                                 gamma=gamma)
        max_diff = jp.max(jp.abs(jp.array(advantages) - ppo_advantages))
        self.assertLess(max_diff, 1e-6)
