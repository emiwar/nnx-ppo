from typing import Any, Optional, Tuple, Dict, Union, Mapping
import enum
import mujoco_playground

import flax.struct
from flax import nnx
import jax

from nnx_ppo.networks.types import PPONetwork, PPONetworkOutput

@flax.struct.dataclass
class TrainingState:
    networks: PPONetwork
    network_states: Any
    env_states: mujoco_playground.State
    optimizer: nnx.Optimizer
    rng_key: jax.Array
    steps_taken: jax.Array

@flax.struct.dataclass
class Transition:
    """Environment state for training and inference."""
    obs: Any
    network_output: PPONetworkOutput
    rewards: Union[Dict, jax.Array]
    done: jax.Array
    truncated: jax.Array
    next_obs: Any
    metrics: Dict[str, Any]

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
    ALL = LOSSES | ACTOR_EXTRA | CRITIC_EXTRA | TRAIN_ROLLOUT_STATS | TRAINING_ENV_METRICS | GRAD_NORM | WEIGHTS | ROLLOUT_OBS
    NONE = 0