"""Test 22: CartpoleSwingup with Brax-matching hyperparameters.

Key changes from Test 21:
- n_epochs: 4 → 16 (Brax default)
- n_minibatches: 8 → 32 (Brax default)
- n_envs: 1024 → 2048 (Brax default)
- Total gradient updates per step: 32 → 512 (16x more!)

Expected: Should approach Brax's ~685-800+ reward
"""
import jax
import jax.numpy as jp
from flax import nnx
import mujoco_playground

from nnx_ppo.networks.feedforward import MLPActorCritic
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.wrappers import episode_wrapper, reward_scaling_wrapper
from nnx_ppo.algorithms import ppo, rollout

SEED = 52
env = mujoco_playground.registry.load("CartpoleSwingup")

# Use Brax's reward scaling
train_env = reward_scaling_wrapper.RewardScalingWrapper(env, 10.0)
train_env = episode_wrapper.EpisodeWrapper(train_env, 1000)

rngs = nnx.Rngs(SEED)
nets = MLPActorCritic(env.observation_size, env.action_size,
                      actor_hidden_sizes=[64, 64, 64],
                      critic_hidden_sizes=[64, 64, 64],
                      rngs=rngs,
                      transfer_function=nnx.swish,
                      action_sampler=NormalTanhSampler(rngs, entropy_weight=1e-2, min_std=1e-3),
                      normalize_obs=True)

config = ppo.default_config()
config.normalize_advantages = False
config.discounting_factor = 0.995

# BRAX-MATCHING HYPERPARAMETERS
config.n_envs = 2048           # Brax: 2048 (was 1024)
config.rollout_length = 30     # Brax: 30
config.n_epochs = 16           # Brax: 16 (was 4) - 4x more!
config.n_minibatches = 32      # Brax: 32 (was 8) - 4x more!

# Total gradient updates per PPO step: 16 * 32 = 512 (was 4 * 8 = 32)

training_state = ppo.new_training_state(train_env, nets, n_envs=config.n_envs, learning_rate=1e-3, seed=SEED)
ppo_step_jit = nnx.jit(ppo.ppo_step, static_argnums=(0, 2, 3, 7, 8, 9, 10, 11))
eval_rollout_jit = nnx.jit(rollout.eval_rollout, static_argnums=(0, 2, 3))

print("=" * 70)
print("TEST 22: CartpoleSwingup with Brax-matching hyperparameters")
print("=" * 70)
print("Key changes from Test 21:")
print("  - n_epochs: 4 → 16 (4x more gradient updates)")
print("  - n_minibatches: 8 → 32 (4x more gradient updates)")
print("  - n_envs: 1024 → 2048 (2x more samples)")
print("  - Total gradient updates per step: 32 → 512 (16x more!)")
print()
print("Config: reward_scaling=10.0, normalize_obs=TRUE, normalize_advantages=False")
print("        actor=[64]*3, critic=[64]*3, entropy_weight=0.01, lr=1e-3")
print("        n_epochs=16, n_minibatches=32, n_envs=2048")
print("Expected: Should approach Brax's ~685-800+ reward")
print("=" * 70)
print()

# Calculate steps per iteration
steps_per_iter = config.n_envs * config.rollout_length
print(f"Steps per iteration: {steps_per_iter:,}")
print(f"Gradient updates per iteration: {config.n_epochs * config.n_minibatches}")
print()
print("Starting training...")

best_reward = 0
for iter in range(500):  # More iterations to match ~30M steps
    training_state, metrics = ppo_step_jit(
        train_env, training_state,
        config.n_envs, config.rollout_length,
        config.gae_lambda, config.discounting_factor,
        config.clip_range, config.normalize_advantages,
        config.n_epochs, config.n_minibatches, ppo.LoggingLevel.ALL,
        (0, 25, 50, 75, 100)
    )

    # Extract key metrics
    critic_loss = metrics['losses/critic']
    actor_loss = metrics['losses/actor']
    r_squared = metrics.get('losses/critic_R^2', 0.0)
    sigma_min = metrics['net/action_sampler/sigma/p0']
    sigma_max = metrics['net/action_sampler/sigma/p100']
    sigma_median = metrics['net/action_sampler/sigma/p50']
    mu_min = metrics['net/action_sampler/mu/p0']
    mu_max = metrics['net/action_sampler/mu/p100']

    # Check for issues
    has_nan = jp.isnan(critic_loss) or jp.isnan(actor_loss)
    has_inf = jp.isinf(critic_loss) or jp.isinf(actor_loss)

    # Log every 10 iterations or first 10
    if iter < 10 or iter % 10 == 0:
        nets.eval()
        eval_metrics = eval_rollout_jit(env, nets, 64, 1000, jax.random.key(SEED))
        nets.train()

        reward = eval_metrics['episode_reward_mean']
        best_reward = max(best_reward, reward)
        total_steps = training_state.steps_taken

        print(f"Iter {iter:3d} | Steps: {total_steps:>10,} | Reward: {reward:7.1f} (best: {best_reward:.1f}) | "
              f"CriticLoss: {critic_loss:8.2f} (R²={r_squared:.3f}) | "
              f"ActorLoss: {actor_loss:7.2f}")
        print(f"         | σ: [{sigma_min:.4f}, {sigma_median:.4f}, {sigma_max:.4f}] | "
              f"μ: [{mu_min:5.2f}, {mu_max:5.2f}]")

    # Early stopping checks
    if has_nan:
        print(f"ERROR at iter {iter}: NaN detected!")
        break
    if has_inf:
        print(f"ERROR at iter {iter}: Inf detected!")
        break
    if mu_max > 20 or mu_min < -20:
        print(f"ERROR at iter {iter}: Policy mean diverging: μ ∈ [{mu_min:.1f}, {mu_max:.1f}]")
        break

    # Success check
    if best_reward > 800:
        print(f"\n*** SUCCESS: Reached {best_reward:.1f} reward! ***")
        # Continue training to see if it stabilizes

print(f"\nTraining completed. Best reward: {best_reward:.1f}")
print(f"Total steps: {training_state.steps_taken:,}")
