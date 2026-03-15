from absl.testing import absltest
import jax.numpy as jp
import numpy as np
from flax import nnx
import mujoco_playground

from nnx_ppo.networks import factories
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.algorithms import rollout, ppo
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.algorithms.config import PPOConfig, EvalConfig, TrainConfig
import nnx_ppo.test_dummies.move_to_center_env
from nnx_ppo.test_dummies.dict_obs_act_env import (
    DictObsActEnv,
    DictObsActNet,
    TwoArmEnv,
    TwoArmNet,
)


class PPOTest(absltest.TestCase):

    def setUp(self):
        SEED = 17
        self.env = mujoco_playground.registry.load("CartpoleBalance")
        self.nets = factories.make_mlp_actor_critic(
            self.env.observation_size,  # type: ignore[arg-type]
            self.env.action_size,
            actor_hidden_sizes=[16, 16],
            critic_hidden_sizes=[16, 16],
            rngs=nnx.Rngs(SEED),
            normalize_obs=True,
        )

    def test_ppo_step(self):
        SEED = 18
        config = ppo.default_config()
        training_state = ppo.new_training_state(
            self.env, self.nets, config.ppo.n_envs, SEED
        )
        self.assertEqual(training_state.steps_taken, 0)
        training_state, metrics = ppo.ppo_step(
            self.env,
            training_state,
            config.ppo.n_envs,
            config.ppo.rollout_length,
            config.ppo.gae_lambda,
            config.ppo.discounting_factor,
            config.ppo.clip_range,
            config.ppo.normalize_advantages,
            config.ppo.combine_advantages,
            config.ppo.n_epochs,
            config.ppo.n_minibatches,
            config.ppo.critic_loss_weight,
            LoggingLevel.ALL,
        )
        self.assertEqual(
            training_state.steps_taken, config.ppo.n_envs * config.ppo.rollout_length
        )
        for k, v in metrics.items():
            self.assertTrue(jp.all(jp.isfinite(v)), f"metrics[{k}] not finite.")

    def test_ppo_step_twice(self):
        SEED = 18
        config = TrainConfig(ppo=PPOConfig(n_envs=4))
        training_state = ppo.new_training_state(
            self.env, self.nets, config.ppo.n_envs, SEED
        )
        self.assertEqual(training_state.steps_taken, 0)
        training_state, metrics = ppo.ppo_step(
            self.env,
            training_state,
            config.ppo.n_envs,
            config.ppo.rollout_length,
            config.ppo.gae_lambda,
            config.ppo.discounting_factor,
            config.ppo.clip_range,
            config.ppo.normalize_advantages,
            config.ppo.combine_advantages,
            config.ppo.n_epochs,
            config.ppo.n_minibatches,
            config.ppo.critic_loss_weight,
            LoggingLevel.ALL,
            (0, 25, 50, 75, 100),
        )
        self.assertEqual(
            training_state.steps_taken, config.ppo.n_envs * config.ppo.rollout_length
        )
        training_state, metrics = ppo.ppo_step(
            self.env,
            training_state,
            config.ppo.n_envs,
            config.ppo.rollout_length,
            config.ppo.gae_lambda,
            config.ppo.discounting_factor,
            config.ppo.clip_range,
            config.ppo.normalize_advantages,
            config.ppo.combine_advantages,
            config.ppo.n_epochs,
            config.ppo.n_minibatches,
            config.ppo.critic_loss_weight,
            LoggingLevel.ALL,
        )
        self.assertEqual(
            training_state.steps_taken,
            config.ppo.n_envs * config.ppo.rollout_length * 2,
        )

    def test_ppo_step_jit(self):
        SEED = 18
        config = ppo.default_config()
        training_state = nnx.jit(ppo.new_training_state, static_argnums=(0, 2, 3))(
            self.env, self.nets, config.ppo.n_envs, SEED
        )
        self.assertEqual(training_state.steps_taken, 0)
        ppo_step_fcn = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3, 7, 8, 9, 10, 11, 12))
        training_state, metrics = ppo_step_fcn(
            self.env,
            training_state,
            config.ppo.n_envs,
            config.ppo.rollout_length,
            config.ppo.gae_lambda,
            config.ppo.discounting_factor,
            config.ppo.clip_range,
            config.ppo.normalize_advantages,
            config.ppo.combine_advantages,
            config.ppo.n_epochs,
            config.ppo.n_minibatches,
            config.ppo.critic_loss_weight,
            LoggingLevel.ALL,
        )
        self.assertEqual(
            training_state.steps_taken, config.ppo.n_envs * config.ppo.rollout_length
        )
        training_state, metrics = ppo_step_fcn(
            self.env,
            training_state,
            config.ppo.n_envs,
            config.ppo.rollout_length,
            config.ppo.gae_lambda,
            config.ppo.discounting_factor,
            config.ppo.clip_range,
            config.ppo.normalize_advantages,
            config.ppo.combine_advantages,
            config.ppo.n_epochs,
            config.ppo.n_minibatches,
            config.ppo.critic_loss_weight,
            LoggingLevel.ALL,
        )
        self.assertEqual(
            training_state.steps_taken,
            config.ppo.n_envs * config.ppo.rollout_length * 2,
        )

    def test_minibatching(self):
        SEED = 18
        config = ppo.default_config()
        training_state = nnx.jit(ppo.new_training_state, static_argnums=(0, 2, 3))(
            self.env, self.nets, config.ppo.n_envs, SEED
        )
        self.assertEqual(training_state.steps_taken, 0)
        ppo_step_fcn = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3, 7, 8, 9, 10, 11, 12))
        training_state, metrics = ppo_step_fcn(
            self.env,
            training_state,
            config.ppo.n_envs,
            config.ppo.rollout_length,
            config.ppo.gae_lambda,
            config.ppo.discounting_factor,
            config.ppo.clip_range,
            config.ppo.normalize_advantages,
            config.ppo.combine_advantages,
            config.ppo.n_epochs,
            config.ppo.n_minibatches,
            config.ppo.critic_loss_weight,
            LoggingLevel.ALL,
        )
        self.assertEqual(
            training_state.steps_taken, config.ppo.n_envs * config.ppo.rollout_length
        )
        training_state, metrics = ppo_step_fcn(
            self.env,
            training_state,
            config.ppo.n_envs,
            config.ppo.rollout_length,
            config.ppo.gae_lambda,
            config.ppo.discounting_factor,
            config.ppo.clip_range,
            config.ppo.normalize_advantages,
            config.ppo.combine_advantages,
            config.ppo.n_epochs,
            config.ppo.n_minibatches,
            config.ppo.critic_loss_weight,
            LoggingLevel.ALL,
        )
        self.assertEqual(
            training_state.steps_taken,
            config.ppo.n_envs * config.ppo.rollout_length * 2,
        )

    def test_rollout_only(self):
        SEED = 18
        config = ppo.default_config()
        training_state = nnx.jit(ppo.new_training_state, static_argnums=(0, 2))(
            self.env, self.nets, config.ppo.n_envs, SEED
        )
        unroll_jit = nnx.jit(rollout.unroll_env, static_argnums=(0, 4))
        unroll_jit(
            self.env,
            training_state.env_states,
            training_state.networks,
            training_state.network_states,
            config.ppo.rollout_length,
            training_state.rng_key,
        )

    def test_train_ppo(self):
        SEED = 21
        config = TrainConfig(
            ppo=PPOConfig(total_steps=512_000),
            eval=EvalConfig(enabled=False),
        )
        result = ppo.train_ppo(self.env, self.nets, config, seed=SEED)
        self.assertEqual(result.total_steps, config.ppo.total_steps)

    def test_gae(self):
        SEED = 23
        N_ENVS = 512
        T_STEPS = 100
        gamma = 0.8
        lambda_ = 0.95

        np.random.seed(SEED)
        rewards = np.random.normal(size=(T_STEPS, N_ENVS))
        values = np.random.normal(size=(T_STEPS + 1, N_ENVS))
        done = np.random.choice([True, False], size=(T_STEPS, N_ENVS), p=[0.01, 0.99])
        truncation = np.random.choice([True, False], size=(T_STEPS, N_ENVS))
        truncation = np.logical_and(done, truncation)

        advantages = np.full((T_STEPS, N_ENVS), np.nan)
        for t in reversed(range(T_STEPS)):
            next_values = values[t + 1, :].copy()
            next_values[done[t, :]] = 0.0
            advantages[t, :] = rewards[t, :] + gamma * next_values - values[t, :]
            advantages[t, truncation[t, :]] = 0.0
            if t < T_STEPS - 1:
                advantages[t, :] += (
                    gamma * lambda_ * advantages[t + 1, :] * (1 - done[t, :])
                )

        ppo_advantages = ppo.gae(
            rewards=jp.array(rewards),
            values_excl_last=jp.array(values[:-1]),
            last_value=jp.array(values[-1]),
            done=jp.array(done, dtype=bool),
            truncation=jp.array(truncation, dtype=bool),
            lambda_=lambda_,
            gamma=gamma,
        )
        max_diff = jp.max(jp.abs(jp.array(advantages) - ppo_advantages))
        self.assertLess(max_diff, 1e-6)

    def test_ppo_move_to_center_env(self):
        SEED = 22
        env = nnx_ppo.test_dummies.move_to_center_env.MoveToCenterEnv(
            reward_falloff=1.0, border_radius=10.0
        )
        nets = factories.make_mlp_actor_critic(
            env.observation_size,
            env.action_size,
            actor_hidden_sizes=[128, 128],
            critic_hidden_sizes=[128, 128],
            rngs=nnx.Rngs(SEED, action_sampling=SEED),
        )
        config = ppo.default_config()
        training_state = ppo.new_training_state(env, nets, config.ppo.n_envs, SEED)
        ppo_step_jit = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3, 7, 8, 9, 10, 11, 12))
        n_updates = 0
        while training_state.steps_taken < config.ppo.total_steps:
            training_state, metrics = ppo_step_jit(
                env,
                training_state,
                config.ppo.n_envs,
                config.ppo.rollout_length,
                config.ppo.gae_lambda,
                config.ppo.discounting_factor,
                config.ppo.clip_range,
                config.ppo.normalize_advantages,
                config.ppo.combine_advantages,
                config.ppo.n_epochs,
                config.ppo.n_minibatches,
                config.ppo.critic_loss_weight,
            )
            n_updates += 1
            self.assertEqual(
                training_state.steps_taken,
                n_updates * config.ppo.rollout_length * config.ppo.n_envs,
            )

            # Arbitrary threshold that empirically has been reasonably easy to reach
            if training_state.steps_taken > 1_500_000:
                self.assertGreater(metrics["episode_reward_mean"].max(), 95.0)

    def test_ppo_normalize_obs_counter(self):
        SEED = 22
        env = nnx_ppo.test_dummies.move_to_center_env.MoveToCenterEnv(
            reward_falloff=1.0, border_radius=10.0
        )
        nets = factories.make_mlp_actor_critic(
            env.observation_size,
            env.action_size,
            actor_hidden_sizes=[64, 64],
            critic_hidden_sizes=[64, 64],
            rngs=nnx.Rngs(SEED, action_sampling=SEED),
            normalize_obs=True,
        )
        config = ppo.default_config()
        training_state = ppo.new_training_state(env, nets, config.ppo.n_envs, SEED)
        ppo_step_jit = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3, 7, 8, 9, 10, 11, 12))
        n_updates = 0
        while training_state.steps_taken < config.ppo.total_steps:
            training_state, metrics = ppo_step_jit(
                env,
                training_state,
                config.ppo.n_envs,
                config.ppo.rollout_length,
                config.ppo.gae_lambda,
                config.ppo.discounting_factor,
                config.ppo.clip_range,
                config.ppo.normalize_advantages,
                config.ppo.combine_advantages,
                config.ppo.n_epochs,
                config.ppo.n_minibatches,
                config.ppo.critic_loss_weight,
            )
            n_updates += 1
            self.assertEqual(
                training_state.steps_taken,
                n_updates * config.ppo.rollout_length * config.ppo.n_envs,
            )
            assert isinstance(nets.preprocessor, Normalizer)
            self.assertEqual(
                nets.preprocessor.counter[...],
                n_updates * config.ppo.rollout_length * config.ppo.n_envs,
            )

            # Arbitrary threshold that empirically has been reasonably easy to reach
            if training_state.steps_taken > 1_500_000:
                self.assertGreater(metrics["episode_reward_mean"].max(), 95.0)


class DictObsActTest(absltest.TestCase):

    def test_ppo_step_dict_obs_act(self):
        """Verify dict obs and dict actions pipe correctly through ppo_step."""
        SEED = 42
        env = DictObsActEnv()
        nets = DictObsActNet(nnx.Rngs(SEED))
        config = PPOConfig(n_envs=8, rollout_length=4, n_epochs=2, n_minibatches=2)
        training_state = ppo.new_training_state(env, nets, config.n_envs, SEED)
        self.assertEqual(training_state.steps_taken, 0)
        training_state, metrics = ppo.ppo_step(
            env,
            training_state,
            config.n_envs,
            config.rollout_length,
            config.gae_lambda,
            config.discounting_factor,
            config.clip_range,
            config.normalize_advantages,
            config.combine_advantages,
            config.n_epochs,
            config.n_minibatches,
            config.critic_loss_weight,
            # Avoid LoggingLevel.ACTOR_EXTRA: it accesses metrics["action_sampler"]
            # which is not provided by DictObsActNet.
            LoggingLevel.LOSSES,
        )
        self.assertEqual(
            training_state.steps_taken, config.n_envs * config.rollout_length
        )
        for k, v in metrics.items():
            self.assertTrue(jp.all(jp.isfinite(v)), f"metrics[{k}] not finite.")

    def test_ppo_step_dict_rewards(self):
        """Verify dict rewards with per-component value estimates pipe correctly through ppo_step."""
        SEED = 43
        env = TwoArmEnv()
        nets = TwoArmNet(nnx.Rngs(SEED))
        config = PPOConfig(n_envs=8, rollout_length=4, n_epochs=2, n_minibatches=2)
        training_state = ppo.new_training_state(env, nets, config.n_envs, SEED)
        self.assertEqual(training_state.steps_taken, 0)
        training_state, metrics = ppo.ppo_step(
            env,
            training_state,
            config.n_envs,
            config.rollout_length,
            config.gae_lambda,
            config.discounting_factor,
            config.clip_range,
            config.normalize_advantages,
            config.combine_advantages,
            config.n_epochs,
            config.n_minibatches,
            config.critic_loss_weight,
            LoggingLevel.LOSSES | LoggingLevel.CRITIC_EXTRA,
        )
        self.assertEqual(
            training_state.steps_taken, config.n_envs * config.rollout_length
        )
        for k, v in metrics.items():
            self.assertTrue(jp.all(jp.isfinite(v)), f"metrics[{k}] not finite.")

    def test_ppo_step_combine_advantages(self):
        """Verify dict rewards with per-component value estimates pipe correctly through ppo_step."""
        SEED = 43
        env = TwoArmEnv()
        nets = TwoArmNet(nnx.Rngs(SEED))
        config = PPOConfig(n_envs=8, rollout_length=4, n_epochs=2, n_minibatches=2)
        config.combine_advantages = True
        training_state = ppo.new_training_state(env, nets, config.n_envs, SEED)
        self.assertEqual(training_state.steps_taken, 0)
        training_state, metrics = ppo.ppo_step(
            env,
            training_state,
            config.n_envs,
            config.rollout_length,
            config.gae_lambda,
            config.discounting_factor,
            config.clip_range,
            config.normalize_advantages,
            config.combine_advantages,
            config.n_epochs,
            config.n_minibatches,
            config.critic_loss_weight,
            LoggingLevel.LOSSES | LoggingLevel.CRITIC_EXTRA,
        )
        self.assertEqual(
            training_state.steps_taken, config.n_envs * config.rollout_length
        )
        for k, v in metrics.items():
            self.assertTrue(jp.all(jp.isfinite(v)), f"metrics[{k}] not finite.")
