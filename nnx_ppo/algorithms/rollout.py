from typing import Any, NamedTuple, Optional
import functools

from flax import nnx
import jax
import jax.numpy as jp
from jaxtyping import Array, Float, Key, Shaped, PRNGKeyArray
import nnx_ppo.networks.types
from nnx_ppo.networks.types import PPONetwork, ModuleState
from nnx_ppo.algorithms.types import Transition, RLEnv, EnvState

def single_transition(
    env: RLEnv,
    networks: PPONetwork,
    carry: tuple[ModuleState, EnvState],
    rng_keys_for_env_reset: Key[Array, "batch"],
) -> tuple[tuple[ModuleState, EnvState], Transition]:
    network_state, env_state = carry
    next_network_state, network_output = networks(network_state, env_state.obs)
    next_env_state = jax.vmap(env.step)(env_state, network_output.actions)
    transition = Transition(
        obs=env_state.obs,
        network_output=network_output,
        rewards=next_env_state.reward,
        done=next_env_state.done.astype(bool),
        truncated=next_env_state.info.get(
            "truncated", jp.zeros(next_env_state.done.shape, bool)
        ).astype(bool),
        next_obs=next_env_state.obs,
        metrics={
            "env": next_env_state.metrics,
            "net": network_output.metrics,
        },
    )

    done = transition.done
    reset_states = jax.vmap(env.reset)(rng_keys_for_env_reset)
    next_env_state = tree_where(done, reset_states, next_env_state)

    reset_network_states = networks.reset_state(next_network_state)
    next_network_state = tree_where(done, reset_network_states, next_network_state)

    return (next_network_state, next_env_state), transition


def unroll_env(
    env: RLEnv,
    env_state: EnvState,
    networks: PPONetwork,
    network_state: ModuleState,
    unroll_length: int,
    rng_key_for_env_reset: PRNGKeyArray,
) -> tuple[ModuleState, EnvState, Transition]:
    batch_size = env_state.done.shape[0]
    rng_keys_for_env_reset = jax.random.split(
        rng_key_for_env_reset, (unroll_length, batch_size)
    )
    step = functools.partial(single_transition, env)
    (final_network_state, final_env_state), rollout = nnx.scan(
        step,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0),
        out_axes=(nnx.Carry, 0),
        length=unroll_length,
    )(networks, (network_state, env_state), rng_keys_for_env_reset)
    shapes_match = jax.tree.map(
        lambda v, r: v.shape == r.shape,
        rollout.network_output.value_estimates,
        rollout.rewards,
    )
    assert all(jax.tree.leaves(shapes_match))
    return final_network_state, final_env_state, rollout


def _add_reward_metrics(
    out: dict,
    name: str,
    reward: Any,
    percentile_levels: Optional[tuple[int, ...]],
) -> None:
    """Recursively build named metrics from a reward PyTree (scalar array or dict)."""
    from collections.abc import Mapping

    if isinstance(reward, Mapping):
        for k, v in reward.items():
            _add_reward_metrics(out, f"{name}/{k}", v, percentile_levels)
    elif percentile_levels is not None:
        percentiles = jp.percentile(reward, jp.array(percentile_levels))
        for pl, p in zip(percentile_levels, percentiles):
            out[f"{name}/p{int(pl)}"] = p
    else:
        out[f"{name}/mean"] = reward.mean()
        out[f"{name}/std"] = reward.std()


def eval_rollout(
    env: RLEnv,
    networks: PPONetwork,
    n_envs: int,
    max_episode_length: int,
    key: PRNGKeyArray,
    logging_percentiles: Optional[tuple[int, ...]] = None,
) -> dict[str, Float[Array, ""]]:
    env_keys = jax.random.split(key, n_envs)
    env_states = jax.vmap(env.reset)(env_keys)
    net_states = networks.initialize_state(n_envs)

    def step(env, networks, carry):
        env_state, network_state, cuml_reward, lifespan = carry
        next_network_state, network_output = networks(network_state, env_state.obs)
        next_env_state = jax.vmap(env.step)(env_state, network_output.actions)
        next_env_state = next_env_state.replace(  # type: ignore[attr-defined]
            done=jp.logical_or(next_env_state.done, env_state.done).astype(float)
        )
        # Only accumulate reward if env was not already done before this step
        reward_this_step = jax.tree.map(
            lambda r: jp.where(env_state.done, jp.zeros_like(r), r),
            next_env_state.reward,
        )
        cuml_reward = jax.tree.map(jp.add, cuml_reward, reward_this_step)
        lifespan += jp.where(next_env_state.done, 0.0, 1.0)
        return next_env_state, next_network_state, cuml_reward, lifespan

    step_partial = functools.partial(step, env)
    step_scan = nnx.scan(
        step_partial,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
        out_axes=nnx.Carry,
        length=max_episode_length,
    )
    init_carry = (
        env_states,
        net_states,
        jax.tree.map(jp.zeros_like, env_states.reward),
        jp.zeros(n_envs),
    )
    _, _, cuml_reward, lifespan = step_scan(networks, init_carry)

    metrics = dict(lifespan_mean=lifespan.mean(), lifespan_std=lifespan.std())
    _add_reward_metrics(metrics, "episode_reward", cuml_reward, logging_percentiles)
    if logging_percentiles is not None:
        percentiles = jp.percentile(lifespan, jp.array(logging_percentiles))
        for pl, p in zip(logging_percentiles, percentiles):
            metrics[f"lifespan/p{int(pl)}"] = p
    return metrics

class SlimData(NamedTuple):
    """Minimal mjx.Data fields needed for rendering."""
    qpos: Any
    qvel: Any
    time: Any


class SlimState(NamedTuple):
    """Minimal env state for rendering, avoiding large mjx.Data contact buffers."""
    data: SlimData
    done: Any
    info: Any
    metrics: Any

def _slim(env_state: EnvState) -> SlimState:
    """Extract only the fields needed for rendering from a full env state.

    Avoids storing large mjx.Data contact/constraint buffers (nconmax, njmax)
    that MuJoCo Warp pre-allocates but that are not needed for rendering.
    """
    return SlimState(
        data=SlimData(
            qpos=env_state.data.qpos,
            qvel=env_state.data.qvel,
            time=env_state.data.time,
        ),
        done=env_state.done,
        info=env_state.info,
        metrics=env_state.metrics,
    )


def eval_rollout_for_render_scan(
    env: RLEnv,
    networks: PPONetwork,
    max_episode_length: int,
    key: PRNGKeyArray,
) -> tuple[SlimState, SlimState, Float[Array, ""]]:
    """JIT-compatible scan-based rollout that returns stacked slim states.

    Returns stacked SlimState (qpos/qvel/time/info/done/metrics only) rather
    than the full env state, avoiding large MuJoCo Warp contact buffers.

    Returns:
      stacked_states: SlimState pytree with leading dimension of max_episode_length.
      final_state: The final environment slim state.
      total_reward: Total reward accumulated during the episode.
    """
    key, key2 = jax.random.split(key)
    env_state = env.reset(key)
    net_state = networks.initialize_state(1)
    net_state = jax.tree.map(lambda x: x[0], net_state)

    def step_fn(networks, carry):
        env_state, net_state, cumulative_reward, already_done, rng = carry

        obs_batched = jax.tree.map(lambda x: x[None], env_state.obs)
        net_state_batched = jax.tree.map(lambda x: x[None], net_state)
        next_net_state, network_output = networks(net_state_batched, obs_batched)
        next_net_state = jax.tree.map(lambda x: x[0], next_net_state)
        action = jax.tree.map(lambda x: x[0], network_output.actions)

        next_env_state = env.step(env_state, action)
        # Only accumulate reward if not already done; sum components for scalar total
        reward_sum = sum(jax.tree.leaves(next_env_state.reward))
        new_cumulative_reward = cumulative_reward + jp.where(
            already_done, 0.0, reward_sum
        )
        new_already_done = jp.logical_or(already_done, next_env_state.done)
        next_env_state = jax.lax.cond(
            next_env_state.done, env.reset, lambda rng: next_env_state, rng
        )
        next_net_state = jax.lax.cond(
            next_env_state.done, networks.reset_state, lambda x: x, next_net_state
        )

        (new_rng,) = jax.random.split(rng, 1)
        return (
            next_env_state,
            next_net_state,
            new_cumulative_reward,
            new_already_done,
            new_rng,
        ), _slim(env_state)

    scan_fn = nnx.scan(
        step_fn,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
        out_axes=(nnx.Carry, 0),
        length=max_episode_length,
    )

    init_carry = (env_state, net_state, jp.array(0.0), jp.array(False), key2)
    (final_env_state, _, total_reward, _, _), stacked_states = scan_fn(
        networks, init_carry
    )

    return stacked_states, _slim(final_env_state), total_reward


def unstack_trajectory(stacked_states, final_state, max_episode_length: int):
    """Convert stacked states from scan to a list for rendering.

    This must be called outside of JIT since it creates a Python list.
    """
    trajectory = [
        jax.tree.map(lambda x: x[i], stacked_states) for i in range(max_episode_length)
    ]
    trajectory.append(final_state)
    return trajectory


def tree_where(cond: Shaped[Array, "batch"], on_true: Any, on_false: Any) -> Any:
    def broadcast_where(x, y):
        if (
            x.shape[0] != cond.shape[0]
        ):  # Hack to handle mujoco-warp data which has some fields that are shared and don't have a batch dimension
            return x
        cond_reshaped = cond.reshape(cond.shape + (1,) * (x.ndim - cond.ndim))
        return jp.where(cond_reshaped, x, y)

    return jax.tree.map(broadcast_where, on_true, on_false)
