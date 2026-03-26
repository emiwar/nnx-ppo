"""Runtime types for the PPO algorithm."""

import dataclasses
from typing import Any, Protocol, runtime_checkable
import enum

import jax
from jaxtyping import Array, Float, Bool, PRNGKeyArray, PyTree, Shaped

from nnx_ppo.networks.types import PPONetworkOutput, ModuleState
from nnx_ppo.jax_dataclass import JaxDataclass



@runtime_checkable
class EnvState(Protocol):
    """Minimal environment state interface.

    Satisfied by mujoco_playground.State and any compatible environment state.
    Fields are declared as read-only properties so that frozen dataclasses
    (which have immutable attributes) satisfy this protocol.
    Note: replace() is intentionally omitted — playground's State has it at
    runtime (via @struct.dataclass) but not in its type stubs.
    """
    @property
    def obs(self) -> Any: ...
    @property
    def done(self) -> Shaped[Array, "..."]: ...  # bool or float depending on env
    @property
    def reward(self) -> Any: ...
    @property
    def info(self) -> dict[str, Any]: ...
    @property
    def metrics(self) -> dict[str, Any]: ...


@runtime_checkable
class RLEnv(Protocol):
    """Minimal RL environment interface.

    Satisfied by mujoco_playground.MjxEnv and any compatible environment.
    """

    def reset(self, rng: PRNGKeyArray) -> EnvState: ...
    def step(self, state: Any, action: Any) -> EnvState: ...


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class TrainingState(JaxDataclass):
    networks: Any  # PPONetwork, but Any because flax.nnx transforms it during JIT
    network_states: Any
    env_states: Any  # EnvState, but Any because the concrete type is mujoco_playground.State
    optimizer: Any  # nnx.Optimizer, but Any because flax.nnx transforms it during JIT
    rng_key: PRNGKeyArray
    steps_taken: Float[Array, ""]


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Transition(JaxDataclass):
    """Environment state for training and inference.

    Note: rewards, done, truncated use *time to allow both [batch] (single timestep)
    and [time, batch] (full rollout) shapes.
    """

    obs: PyTree[Float[Array, "..."]]
    network_output: PPONetworkOutput
    rewards: PyTree[Float[Array, "*time batch"]]
    done: Bool[Array, "*time batch"]
    truncated: Bool[Array, "*time batch"]
    next_obs: PyTree[Float[Array, "..."]]  # Same pytree structure as obs
    metrics: dict[str, Any]


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class DistillationTransition(JaxDataclass):
    """Rollout transition for distillation training.

    Carries both student and teacher network outputs. The student's actions
    drive the environment; the teacher's outputs (pre-computed outside of
    gradient computation) serve as the distillation target.
    """

    obs: PyTree[Float[Array, "..."]]
    student_output: PPONetworkOutput  # drives env; used in student time-scan replay
    teacher_output: PPONetworkOutput  # frozen reference; raw_actions = teacher mean (eval mode)
    rewards: PyTree[Float[Array, "*time batch"]]
    done: Bool[Array, "*time batch"]
    truncated: Bool[Array, "*time batch"]
    next_obs: PyTree[Float[Array, "..."]]
    metrics: dict[str, Any]


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class DistillationState(JaxDataclass):
    """Training state for distillation.

    Mirrors TrainingState but has separate student and teacher states.
    The teacher is passed as an external argument (like env) and is not
    stored here. Only the teacher's per-env carry state is tracked.
    """

    student: Any  # PPONetwork, but Any because flax.nnx transforms it during JIT
    student_states: Any  # ModuleState [n_envs, ...]
    teacher_states: Any  # ModuleState [n_envs, ...] (tracked but not trained)
    env_states: Any  # EnvState
    optimizer: Any  # nnx.Optimizer (over student only)
    rng_key: PRNGKeyArray
    steps_taken: Float[Array, ""]


class LoggingLevel(enum.Flag):
    LOSSES = enum.auto()
    CRITIC_EXTRA = enum.auto()
    ACTOR_EXTRA = enum.auto()
    TRAIN_ROLLOUT_STATS = enum.auto()
    ROLLOUT_OBS = enum.auto()
    TRAINING_ENV_METRICS = enum.auto()
    GRAD_NORM = enum.auto()
    WEIGHTS = enum.auto()
    BASIC = LOSSES
    ALL = (
        LOSSES
        | ACTOR_EXTRA
        | CRITIC_EXTRA
        | TRAIN_ROLLOUT_STATS
        | TRAINING_ENV_METRICS
        | GRAD_NORM
        | WEIGHTS
        | ROLLOUT_OBS
    )
    NONE = 0