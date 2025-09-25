import functools
from typing import Tuple, Any, Dict

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jp
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
        training_state = ppo.ppo_step(self.env, training_state, config.n_envs, config.rollout_length,
                                      config.gae_lambda, config.discounting_factor, config.clip_range)
        self.assertEquals(training_state.steps_taken, config.n_envs*config.rollout_length)

    def test_ppo_step_twice(self):
        SEED = 18
        config = ppo.default_config()
        training_state = ppo.new_training_state(self.env, self.nets, config.n_envs, SEED)
        self.assertEquals(training_state.steps_taken, 0)
        training_state = ppo.ppo_step(self.env, training_state, config.n_envs, config.rollout_length,
                                      config.gae_lambda, config.discounting_factor, config.clip_range)
        self.assertEquals(training_state.steps_taken, config.n_envs*config.rollout_length)
        training_state = ppo.ppo_step(self.env, training_state, config.n_envs, config.rollout_length,
                                      config.gae_lambda, config.discounting_factor, config.clip_range)
        self.assertEquals(training_state.steps_taken, config.n_envs*config.rollout_length*2)

    def test_ppo_step_jit(self):
        SEED = 18
        config = ppo.default_config()
        training_state = nnx.jit(ppo.new_training_state, static_argnums=(0, 2, 3))(self.env, self.nets, config.n_envs, SEED)
        self.assertEquals(training_state.steps_taken, 0)
        #ppo_step_fcn = functools.partial(ppo.ppo_step, self.env)
        ppo_step_fcn = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3))
        #ppo_step_fcn = checkify.checkify(ppo_step_fcn)
        training_state = ppo_step_fcn(self.env, training_state, config.n_envs, config.rollout_length,
                                      config.gae_lambda, config.discounting_factor, config.clip_range)
        #err.throw()
        self.assertEquals(training_state.steps_taken, config.n_envs*config.rollout_length)
        training_state = ppo_step_fcn(self.env, training_state, config.n_envs, config.rollout_length,
                                      config.gae_lambda, config.discounting_factor, config.clip_range)
        #err.throw()
        self.assertEquals(training_state.steps_taken, config.n_envs*config.rollout_length*2)


    def test_rollout_only(self):
        def unroll(env: mujoco_playground.MjxEnv,
                   training_state: ppo.TrainingState,
                   n_envs: int, rollout_length: int):
            reset_keys = jax.random.split(training_state.rng_key, n_envs)

            unroll_vmap = nnx.vmap(rollout.unroll_env,
                                in_axes  = (None, 0, None, 0, None, 0),
                                out_axes = (0, 0, 0))
            unroll_vmap = nnx.split_rngs(splits=n_envs)(unroll_vmap)
            _, _, rollout_data = unroll_vmap(
                env,
                training_state.env_states,
                training_state.networks,
                training_state.network_states,
                rollout_length,
                reset_keys
            )
            return rollout_data
        
        SEED = 18
        config = ppo.default_config()
        training_state = nnx.jit(ppo.new_training_state, static_argnums=(0, 2))(self.env, self.nets, config.n_envs, SEED)
        unroll_jit = nnx.jit(unroll, static_argnums=(0, 2, 3))
        unroll_jit(self.env, training_state, config.n_envs, config.rollout_length)

    def test_ppo_full_loop(self):
        SEED = 21
        ppo.train_ppo(self.env, self.nets, ppo.default_config(), SEED)