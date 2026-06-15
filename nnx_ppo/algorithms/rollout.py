from typing import Any, NamedTuple, Optional
from collections.abc import Mapping
import functools

from flax import nnx
import jax
import jax.numpy as jp
from jaxtyping import Array, Float, Key, Shaped, PRNGKeyArray
from nnx_ppo.networks.types import ModuleState, StatefulModule
from nnx_ppo.algorithms.types import Transition, RLEnv, EnvState, LoggingLevel

def single_transition(
    env: RLEnv,
    networks: StatefulModule,
    carry: tuple[ModuleState, EnvState],
    rng_keys_for_env_reset: Key[Array, "batch"],
) -> tuple[tuple[ModuleState, EnvState], Transition]:
    network_state, env_state = carry
    out = networks(network_state, env_state.obs)
    next_network_state = out.next_state
    ppo_output = out.output
    next_env_state = jax.vmap(env.step)(env_state, ppo_output.actions)
    transition = Transition(
        obs=env_state.obs,
        network_output=ppo_output,
        rewards=next_env_state.reward,
        done=next_env_state.done.astype(bool),
        truncated=next_env_state.info.get(
            "truncated", jp.zeros(next_env_state.done.shape, bool)
        ).astype(bool),
        next_obs=next_env_state.obs,
        metrics={
            "env": next_env_state.metrics,
            "net": out.metrics,
        },
        rollout_extras=out.rollout_extras,
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
    networks: StatefulModule,
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


def _add_distribution_metrics(
    out: dict,
    name: str,
    value: Any,
    percentile_levels: Optional[tuple[int, ...]],
) -> None:
    """Recursively build named metrics from a metric PyTree (scalar array or dict).

    For a dict, recurses per key. For a leaf array it emits ``<name>/p{N}`` when
    ``percentile_levels`` is given, otherwise ``<name>/mean`` and ``<name>/std``.
    """
    if isinstance(value, Mapping):
        for k, v in value.items():
            _add_distribution_metrics(out, f"{name}/{k}", v, percentile_levels)
    elif percentile_levels is not None:
        percentiles = jp.percentile(value, jp.array(percentile_levels))
        for pl, p in zip(percentile_levels, percentiles):
            out[f"{name}/p{int(pl)}"] = p
    else:
        out[f"{name}/mean"] = value.mean()
        out[f"{name}/std"] = value.std()


def _mask_done(done: Shaped[Array, "batch"], x: Float[Array, "batch ..."]) -> Any:
    """Zero out the per-env leaves whose env was already done this step."""
    d = done.reshape(done.shape + (1,) * (x.ndim - done.ndim))
    return jp.where(d, jp.zeros_like(x), x)


def eval_rollout(
    env: RLEnv,
    networks: StatefulModule,
    n_envs: int,
    max_episode_length: int,
    key: PRNGKeyArray,
    logging_percentiles: Optional[tuple[int, ...]] = None,
    logging_level: LoggingLevel = LoggingLevel.NONE,
) -> dict[str, Float[Array, ""]]:
    env_keys = jax.random.split(key, n_envs)
    env_states = jax.vmap(env.reset)(env_keys)
    # The scan latches done as float (below); the initial state must match that
    # dtype or the scan carry types diverge (envs whose reset done is bool).
    env_states = env_states.replace(done=env_states.done.astype(float))
    net_states = networks.initialize_state(n_envs)

    # Env metrics (env_state.metrics) and network module metrics (entropy, KL,
    # forward-model MSE, mu/sigma, ...) are accumulated only when their flag is
    # set. Each per-step leaf is masked by the pre-step done flag and later
    # normalised by per-env lifespan, mirroring the reward accounting. (Full-obs
    # logging via ROLLOUT_OBS is a training-only debug aid; an episode-averaged
    # obs is not meaningful at eval, so it is intentionally not logged here.)
    log_env_metrics = LoggingLevel.ENV_METRICS in logging_level
    log_net_metrics = LoggingLevel.NETWORK_METRICS in logging_level

    def step(env, networks, carry):
        env_state, network_state, cuml_reward, lifespan, accum = carry
        out = networks(network_state, env_state.obs)
        next_network_state = out.next_state
        network_output = out.output
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
        step_metrics = {}
        if log_net_metrics:
            step_metrics["net"] = out.metrics
        if log_env_metrics:
            step_metrics["env"] = next_env_state.metrics
        accum = jax.tree.map(
            lambda c, m: c + _mask_done(env_state.done, m), accum, step_metrics
        )
        return next_env_state, next_network_state, cuml_reward, lifespan, accum

    init_accum: dict[str, Any] = {}
    if log_net_metrics:
        # Probe the metric tree structure to seed a zeroed accumulator.
        probe = networks(net_states, env_states.obs)
        init_accum["net"] = jax.tree.map(jp.zeros_like, probe.metrics)
    if log_env_metrics:
        init_accum["env"] = jax.tree.map(jp.zeros_like, env_states.metrics)

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
        init_accum,
    )
    _, _, cuml_reward, lifespan, accum = step_scan(networks, init_carry)

    # All eval metrics live under ``eval/`` so they never collide with training
    # metrics when both are merged into one dict (train_ppo: metrics.update).
    metrics: dict[str, Any] = {}

    # Headline + "did an eval run?" sentinel: the total episode return (summed
    # across reward keys, the quantity PPO optimises), mean/std over envs. Always
    # emitted, independent of logging_level / logging_percentiles.
    total_return = jax.tree.reduce(jp.add, cuml_reward)
    metrics["eval/episode_reward/mean"] = total_return.mean()
    metrics["eval/episode_reward/std"] = total_return.std()
    if logging_percentiles is not None:
        percentiles = jp.percentile(total_return, jp.array(logging_percentiles))
        for pl, p in zip(logging_percentiles, percentiles):
            metrics[f"eval/episode_reward/p{int(pl)}"] = p
    # Per-term breakdown only for multi-reward (dict) envs; for a scalar reward
    # the headline above is already the full picture.
    if isinstance(cuml_reward, Mapping):
        _add_distribution_metrics(
            metrics, "eval/episode_reward", cuml_reward, logging_percentiles
        )

    # Lifespan follows the normal percentile convention (mean/std, or p{N}).
    _add_distribution_metrics(metrics, "eval/lifespan", lifespan, logging_percentiles)
    if log_net_metrics or log_env_metrics:
        # Local import avoids a circular dependency (metrics imports rollout).
        from nnx_ppo.algorithms.metrics import _log_metric

        denom = jp.maximum(lifespan, 1.0)
        mean_accum = jax.tree.map(
            lambda s: s / denom.reshape(denom.shape + (1,) * (s.ndim - 1)),
            accum,
        )
        if log_net_metrics:
            _log_metric(metrics, "eval/net", mean_accum["net"], logging_percentiles)
        if log_env_metrics:
            _log_metric(metrics, "eval/env", mean_accum["env"], logging_percentiles)
    return metrics

class SlimData(NamedTuple):
    """Minimal mjx.Data fields needed for rendering."""
    qpos: Any
    qvel: Any
    time: Any
    mocap_pos: Any
    mocap_quat: Any
    xfrc_applied: Any


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
            mocap_pos=env_state.data.mocap_pos,
            mocap_quat=env_state.data.mocap_quat,
            xfrc_applied=env_state.data.xfrc_applied,
        ),
        done=env_state.done,
        info=env_state.info,
        metrics=env_state.metrics,
    )


def eval_rollout_for_render_scan(
    env: RLEnv,
    networks: StatefulModule,
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
        out = networks(net_state_batched, obs_batched)
        next_net_state = out.next_state
        network_output = out.output
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
