import functools

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jp
from flax import nnx
import mujoco_playground

from nnx_ppo.networks import factories
from nnx_ppo.algorithms.rollout import (
    unroll_env,
    eval_rollout,
    record_activations_rollout,
)
from nnx_ppo.networks.recording import with_recording, extract_activations
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.test_dummies import dummy_counter
from nnx_ppo.test_dummies import stateful_nets, parrot_env, move_to_center_env
from nnx_ppo.test_dummies.dict_obs_act_env import TwoArmEnv, TwoArmNet


class RolloutTest(absltest.TestCase):

    def setUp(self):
        SEED = 17

        self.env = mujoco_playground.registry.load(
            "CartpoleSwingup", config_overrides={"impl": "jax"}
        )
        self.nets = factories.make_mlp_actor_critic(
            self.env.observation_size,  # type: ignore[arg-type]
            self.env.action_size,
            actor_hidden_sizes=[16, 16],
            critic_hidden_sizes=[16, 16],
            rngs=nnx.Rngs(SEED, action_sampling=SEED),
        )

    def test_single_env_rollout(self):
        N_STEPS = 24
        key = jax.random.key(seed=18)
        env_key, reset_key = jax.random.split(key)
        net_state = self.nets.initialize_state(batch_size=1)
        env_state = jax.vmap(self.env.reset)(jax.random.split(env_key, 1))

        next_net_state, next_env_state, rollout_data = unroll_env(
            self.env, env_state, self.nets, net_state, N_STEPS, reset_key
        )
        self.assertEqual(rollout_data.done.shape, (N_STEPS, 1))
        self.assertEqual(rollout_data.rewards.shape, (N_STEPS, 1))
        self.assertEqual(
            rollout_data.obs.shape, (N_STEPS, 1, self.env.observation_size)
        )
        self.assertEqual(
            rollout_data.next_obs.shape, (N_STEPS, 1, self.env.observation_size)
        )
        self.assertEqual(
            rollout_data.network_output.actions.shape,
            (N_STEPS, 1, self.env.action_size),
        )
        self.assertEqual(rollout_data.network_output.loglikelihoods.shape, (N_STEPS, 1))

    def test_single_env_rollout_jit(self):
        N_STEPS = 24
        key = jax.random.key(seed=18)
        env_key, reset_key = jax.random.split(key)
        net_state = self.nets.initialize_state(batch_size=1)
        env_state = jax.vmap(self.env.reset)(jax.random.split(env_key, 1))

        unroll_env_jit = nnx.jit(
            functools.partial(unroll_env, self.env), static_argnames=("unroll_length")
        )
        next_net_state, next_env_state, rollout_data = unroll_env_jit(
            env_state, self.nets, net_state, N_STEPS, reset_key
        )
        self.assertEqual(rollout_data.done.shape, (N_STEPS, 1))
        self.assertEqual(rollout_data.rewards.shape, (N_STEPS, 1))
        self.assertEqual(
            rollout_data.obs.shape, (N_STEPS, 1, self.env.observation_size)
        )
        self.assertEqual(
            rollout_data.next_obs.shape, (N_STEPS, 1, self.env.observation_size)
        )
        self.assertEqual(
            rollout_data.network_output.actions.shape,
            (N_STEPS, 1, self.env.action_size),
        )
        self.assertEqual(rollout_data.network_output.loglikelihoods.shape, (N_STEPS, 1))

        # Do another rollout just to be sure there're no stray tracers
        unroll_env_jit(next_env_state, self.nets, next_net_state, N_STEPS, reset_key)

    def test_batch_env_rollout(self):
        N_ENVS = 256
        N_STEPS = 24
        key = jax.random.key(seed=18)
        env_key, reset_key = jax.random.split(key)
        net_states = self.nets.initialize_state(batch_size=N_ENVS)
        env_states = jax.vmap(self.env.reset)(jax.random.split(env_key, N_ENVS))

        next_net_state, next_env_state, rollout_data = unroll_env(
            self.env, env_states, self.nets, net_states, N_STEPS, reset_key
        )
        self.assertEqual(rollout_data.done.shape, (N_STEPS, N_ENVS))
        self.assertEqual(rollout_data.rewards.shape, (N_STEPS, N_ENVS))
        self.assertEqual(
            rollout_data.obs.shape, (N_STEPS, N_ENVS, self.env.observation_size)
        )
        self.assertEqual(
            rollout_data.next_obs.shape, (N_STEPS, N_ENVS, self.env.observation_size)
        )
        self.assertEqual(
            rollout_data.network_output.actions.shape,
            (N_STEPS, N_ENVS, self.env.action_size),
        )
        self.assertEqual(
            rollout_data.network_output.loglikelihoods.shape, (N_STEPS, N_ENVS)
        )
        jax.tree.map(
            lambda a, b: self.assertEqual(a.shape, b.shape), net_states, next_net_state
        )
        jax.tree.map(
            lambda a, b: self.assertEqual(a.shape, b.shape), env_states, next_env_state
        )
        unroll_env(
            self.env, next_env_state, self.nets, next_net_state, N_STEPS, reset_key
        )

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
            dummy_env, env_states, dummy_nets, net_state, N_STEPS, reset_key
        )
        self.assertEqual(rollout_data.done.shape, (N_STEPS, 1))
        self.assertEqual(rollout_data.rewards.shape, (N_STEPS, 1))
        self.assertEqual(rollout_data.network_output.loglikelihoods.shape, (N_STEPS, 1))

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
            dummy_env, env_state, dummy_nets, net_state, N_STEPS, reset_key
        )
        self.assertEqual(rollout_data.done.shape, (N_STEPS, 1))
        self.assertEqual(rollout_data.rewards.shape, (N_STEPS, 1))
        self.assertEqual(rollout_data.network_output.loglikelihoods.shape, (N_STEPS, 1))

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
            dummy_env, env_states, dummy_nets, net_states, N_STEPS, reset_key
        )
        self.assertEqual(rollout_data.done.shape, (N_STEPS, N_ENVS))
        self.assertEqual(rollout_data.rewards.shape, (N_STEPS, N_ENVS))
        self.assertEqual(
            rollout_data.network_output.loglikelihoods.shape, (N_STEPS, N_ENVS)
        )

        self.assertGreaterEqual(jp.sum(rollout_data.done), 2 * N_ENVS)
        self.assertLess(jp.sum(rollout_data.done), 10 * N_ENVS)
        self.assertEqual(jp.sum(rollout_data.rewards), N_STEPS * N_ENVS)
        jax.tree.map(
            lambda a, b: self.assertEqual(a.shape, b.shape), net_states, next_net_state
        )
        jax.tree.map(
            lambda a, b: self.assertEqual(a.shape, b.shape), env_states, next_env_state
        )

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
            env, env_state, net, net_state, N_STEPS, reset_key
        )
        self.assertEqual(net.n_calls[...], N_STEPS * N_ENVS)

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
            env, env_state, net, net_state, N_STEPS, reset_key
        )
        self.assertEqual(net.n_calls[...], N_STEPS * N_ENVS)

    def test_eval_rollout_basic_omits_net_metrics(self):
        metrics = eval_rollout(
            self.env, self.nets, n_envs=4, max_episode_length=10,
            key=jax.random.key(0), logging_level=LoggingLevel.BASIC,
        )
        self.assertIn("eval/episode_reward/mean", metrics)
        self.assertIn("eval/episode_reward/std", metrics)
        self.assertIn("eval/lifespan/mean", metrics)
        self.assertIn("eval/lifespan/std", metrics)
        # Every eval metric is eval/-prefixed.
        self.assertTrue(all(k.startswith("eval/") for k in metrics))
        self.assertFalse(any(k.startswith("eval/net") for k in metrics))
        self.assertFalse(any(k.startswith("eval/env") for k in metrics))

    def test_eval_rollout_logs_net_metrics(self):
        metrics = eval_rollout(
            self.env, self.nets, n_envs=4, max_episode_length=10,
            key=jax.random.key(0),
            logging_level=LoggingLevel.BASIC | LoggingLevel.NETWORK_METRICS,
        )
        net_keys = [k for k in metrics if k.startswith("eval/net")]
        self.assertTrue(net_keys, "expected eval/net/* network metrics")
        # NormalTanhSampler emits mu/sigma; they must reach the eval log.
        self.assertTrue(any("mu" in k for k in net_keys))
        for k in net_keys:
            self.assertEqual(jp.asarray(metrics[k]).shape, ())
        # NETWORK_METRICS alone must not pull in env metrics.
        self.assertFalse(any(k.startswith("eval/env") for k in metrics))

    def test_eval_rollout_logs_env_metrics(self):
        metrics = eval_rollout(
            self.env, self.nets, n_envs=4, max_episode_length=10,
            key=jax.random.key(0),
            logging_level=LoggingLevel.BASIC | LoggingLevel.ENV_METRICS,
        )
        env_keys = [k for k in metrics if k.startswith("eval/env")]
        self.assertTrue(env_keys, "expected eval/env/* env metrics")
        self.assertFalse(any(k.startswith("eval/net") for k in metrics))

    def test_eval_rollout_sums_dict_rewards(self):
        """For a dict-reward env the headline is the sum across reward keys,
        and each term also gets its own subtree."""
        env = TwoArmEnv()
        nets = TwoArmNet(nnx.Rngs(0))
        metrics = eval_rollout(
            env, nets, n_envs=4, max_episode_length=8,
            key=jax.random.key(0), logging_level=LoggingLevel.BASIC,
        )
        # Headline aggregate is always present...
        self.assertIn("eval/episode_reward/mean", metrics)
        # ...and each reward term expands to its own subtree.
        self.assertIn("eval/episode_reward/arm1/mean", metrics)
        self.assertIn("eval/episode_reward/arm2/mean", metrics)
        # mean of the summed return == sum of the per-term means.
        self.assertAlmostEqual(
            float(metrics["eval/episode_reward/mean"]),
            float(metrics["eval/episode_reward/arm1/mean"])
            + float(metrics["eval/episode_reward/arm2/mean"]),
            places=4,
        )


class RecordActivationsRolloutTest(absltest.TestCase):

    def setUp(self):
        self.env = mujoco_playground.registry.load(
            "CartpoleSwingup", config_overrides={"impl": "jax"}
        )
        self.nets = factories.make_mlp_actor_critic(
            self.env.observation_size,  # type: ignore[arg-type]
            self.env.action_size,
            actor_hidden_sizes=[16, 16],
            critic_hidden_sizes=[16, 16],
            rngs=nnx.Rngs(0, action_sampling=0),
        )

    def test_returns_stacked_activations_and_dones(self):
        T, N = 6, 3
        acts, dones = record_activations_rollout(
            self.env, self.nets, n_envs=N, max_episode_length=T,
            key=jax.random.key(0),
        )
        self.assertEqual(dones.shape, (T, N))
        # Every recorded leaf has leading dims [T, N, ...].
        leaves = jax.tree.leaves(acts)
        self.assertGreater(len(leaves), 0)
        for leaf in leaves:
            self.assertEqual(leaf.shape[:2], (T, N))
        # The original network was not mutated into a recording one.
        self.assertNotIsInstance(self.nets[0], type(with_recording(self.nets)[0]))

    def test_first_step_matches_manual_call(self):
        T, N = 4, 2
        key = jax.random.key(7)
        acts, _ = record_activations_rollout(
            self.env, self.nets, n_envs=N, max_episode_length=T, key=key,
        )
        # Reproduce the rollout's first step by hand (same reset keys, eval mode).
        rec_net = with_recording(self.nets)
        rec_net.eval()
        env_states = jax.vmap(self.env.reset)(jax.random.split(key, N))
        net_states = rec_net.initialize_state(N)
        out = rec_net(net_states, env_states.obs)
        manual = extract_activations(out.metrics)
        # Loose tolerance: the rollout runs jit/scan-compiled while the manual
        # call is eager, so matmuls differ by float noise (~1e-5).
        match = jax.tree.all(
            jax.tree.map(
                lambda s, m: bool(jp.allclose(s[0], m, atol=1e-3)), acts, manual
            )
        )
        self.assertTrue(match)
