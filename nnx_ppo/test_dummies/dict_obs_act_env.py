"""Dummy environment and network with dict observations and dict actions.

Used to verify that the PPO pipeline handles PyTree obs/actions correctly.
"""

import dataclasses
from typing import Any

import jax
import jax.flatten_util
import jax.numpy as jp
from flax import nnx

from nnx_ppo.jax_dataclass import JaxDataclass
from nnx_ppo.networks.types import (
    ModuleState,
    PPONetworkOutput,
    StatefulModule,
    StatefulModuleOutput,
)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class DictEnvState(JaxDataclass):
    """Environment state with dict obs and optionally dict reward."""

    obs: Any    # {"pos": array[2], "vel": array[2]}
    reward: Any  # jax.Array or dict thereof for multi-reward environments
    done: jax.Array
    info: dict[str, Any]
    metrics: dict[str, Any]


class DictObsActEnv:
    """Simple 2D environment where observations and actions are dicts.

    obs  = {"pos": array[2], "vel": array[2]}
    action = {"force": array[2]}

    Dynamics: vel += force * 0.1, pos += vel
    Reward: exp(-|pos|) — rewarded for staying near origin
    Done:  |pos| > 3.0
    """

    def reset(self, rng: jax.Array) -> DictEnvState:
        pos = jax.random.uniform(rng, (2,), minval=-1.0, maxval=1.0)
        vel = jp.zeros(2)
        return self._make_state(pos, vel)

    def step(self, state: DictEnvState, action: dict) -> DictEnvState:
        new_vel = state.obs["vel"] + action["force"] * 0.1
        new_pos = state.obs["pos"] + new_vel
        return self._make_state(new_pos, new_vel)

    def _make_state(self, pos, vel) -> DictEnvState:
        dist = jp.sqrt(jp.sum(pos**2))
        reward = jp.exp(-dist)
        done = (dist > 3.0).astype(float)
        return DictEnvState(
            obs={"pos": pos, "vel": vel},
            reward=reward,
            done=done,
            info={},
            metrics={},
        )


class DictObsActNet(StatefulModule):
    """Minimal network that consumes dict obs and produces dict actions.

    obs    = {"pos": array[batch, 2], "vel": array[batch, 2]}
    actions = {"force": array[batch, 2]}

    Loglikelihoods are held constant at 0 (sufficient for testing the
    data pipeline; critic gradients still flow).
    """

    def __init__(self, rngs: nnx.Rngs):
        # 4 inputs (pos[2] + vel[2]) -> 2 action dims
        self.actor = nnx.Linear(4, 2, rngs=rngs)
        # 4 inputs -> scalar value estimate
        self.critic = nnx.Linear(4, 1, rngs=rngs)

    def __call__(
        self,
        network_state,
        obs,
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        obs_flat = jp.concatenate([obs["pos"], obs["vel"]], axis=-1)  # [batch, 4]
        actor_out = self.actor(obs_flat)  # [batch, 2]
        value = jp.squeeze(self.critic(obs_flat), axis=-1)  # [batch]

        if rollout_extras is not None:
            raw_action = rollout_extras
        else:
            raw_action = {"force": actor_out}

        actions = {"force": jp.tanh(raw_action["force"])}
        batch_size = obs_flat.shape[0]
        loglikelihoods = jp.zeros(batch_size)

        return StatefulModuleOutput(
            next_state=network_state,
            output=PPONetworkOutput(
                actions=actions,
                loglikelihoods=loglikelihoods,
                value_estimates=value,
            ),
            regularization_loss=jp.zeros(batch_size),
            metrics={},
            rollout_extras=raw_action,
        )

    def initialize_state(self, batch_size: int):
        return ()


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class TwoArmState(JaxDataclass):
    """Environment state with dict obs and optionally dict reward."""
    obs: Any
    reward: Any
    done: jax.Array
    info: dict[str, Any]
    metrics: dict[str, Any]

class TwoArmEnv:
    """Minimal multi-agent env.

    obs    = {"arm1": {"pos": array[2], "vel": array[2]},
              "arm2": {"pos": array[2], "vel": array[2]},}
    action = {"arm1": array[2], "arm2": array[2]}
    reward = {"arm1": Float, "arm2": Float}
    """

    def reset(self, rng: jax.Array) -> TwoArmState:
        pos = {
            "arm1": jax.random.uniform(rng, (2,), minval=-1.0, maxval=1.0),
            "arm2": jax.random.uniform(rng, (2,), minval=-1.0, maxval=1.0),
        }
        vel = {
            "arm1": jp.zeros(2),
            "arm2": jp.zeros(2),
        }
        reward = jax.tree.map(lambda p: jp.exp(-jp.sqrt(jp.sum(p**2))), pos)

        done = jp.logical_or(jp.sqrt(jp.sum(pos["arm1"]**2)) > 3.0,
                             jp.sqrt(jp.sum(pos["arm2"]**2)) > 3.0)
        obs = jax.tree.map(lambda p,v: {"pos": p, "vel": v}, pos, vel)
        return TwoArmState(obs=obs, reward=reward, done=done, info={}, metrics={})

    def step(self, state: TwoArmState, action: dict) -> TwoArmState:
        new_vel = {
            "arm1": state.obs["arm1"]["vel"] + 0.1*action["arm1"],
            "arm2": state.obs["arm2"]["vel"] + 0.1*action["arm2"],
        }
        new_pos = {
            "arm1": state.obs["arm1"]["pos"] + 0.1*new_vel["arm1"],
            "arm2": state.obs["arm2"]["pos"] + 0.1*new_vel["arm2"],
        }
        reward = jax.tree.map(lambda p: jp.exp(-jp.sqrt(jp.sum(p**2))), new_pos)

        done = jp.logical_or(jp.sqrt(jp.sum(new_pos["arm1"]**2)) > 3.0,
                             jp.sqrt(jp.sum(new_pos["arm2"]**2)) > 3.0)
        obs = jax.tree.map(lambda p,v: {"pos": p, "vel": v}, new_pos, new_vel)
        return TwoArmState(obs=obs, reward=reward, done=done, info={}, metrics={})


class TwoArmNet(StatefulModule):
    """Network with dict obs/actions and dict value estimates matching TwoArmState."""

    def __init__(self, rngs: nnx.Rngs):
        self.actor = nnx.Linear(8, 4, rngs=rngs)
        self.critic = nnx.Linear(8, 2, rngs=rngs)

    def __call__(
        self,
        network_state: tuple,
        obs,
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        obs_flat = jax.vmap(lambda t: jax.flatten_util.ravel_pytree(t)[0])(obs)  # [batch, 8]
        actor_out = self.actor(obs_flat)  # [batch, 4]
        critic_out = self.critic(obs_flat)  # [batch, 2]

        actions = {"arm1": actor_out[:, :2], "arm2": actor_out[:, 2:]}
        values = {"arm1": critic_out[:, 0], "arm2": critic_out[:, 1]}
        batch_size = obs_flat.shape[0]

        return StatefulModuleOutput(
            next_state=network_state,
            output=PPONetworkOutput(
                actions=actions,
                loglikelihoods={
                    "arm1": jp.zeros(batch_size),
                    "arm2": jp.zeros(batch_size),
                },
                value_estimates=values,
            ),
            regularization_loss=jp.zeros(batch_size),
            metrics={},
            rollout_extras=None,
        )

    def initialize_state(self, batch_size: int) -> ModuleState:
        return ()
