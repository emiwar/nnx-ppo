"""Configuration dataclasses for train_ppo."""

from typing import Any, Optional, Tuple, Dict, List
from dataclasses import dataclass, field

import numpy as np

from nnx_ppo.algorithms.types import TrainingState, LoggingLevel


@dataclass
class PPOConfig:
    """Core PPO algorithm parameters."""

    n_envs: int = 256
    rollout_length: int = 20
    total_steps: int = 512_000
    gae_lambda: float = 0.95
    discounting_factor: float = 0.99
    clip_range: float = 0.2
    learning_rate: float = 1e-4
    normalize_advantages: bool = True
    n_epochs: int = 4
    n_minibatches: int = 4
    gradient_clipping: Optional[float] = None
    weight_decay: Optional[float] = None
    logging_level: LoggingLevel = LoggingLevel.LOSSES
    logging_percentiles: Optional[Tuple[int, ...]] = None


@dataclass
class EvalConfig:
    """Evaluation rollout configuration."""

    enabled: bool = True
    every_steps: int = 50_000
    n_envs: int = 64
    max_episode_length: int = 1000
    logging_level: LoggingLevel = LoggingLevel.BASIC
    logging_percentiles: Optional[Tuple[int, ...]] = (0, 25, 50, 75, 100)


@dataclass
class VideoConfig:
    """Video recording configuration."""

    enabled: bool = False
    every_steps: int = 200_000
    episode_length: int = 1000
    render_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "height": 480,
            "width": 640,
        }
    )


@dataclass
class TrainConfig:
    """Complete training configuration."""

    ppo: PPOConfig = field(default_factory=PPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    seed: int = 17


@dataclass
class VideoData:
    """Data passed to video callback."""

    frames: np.ndarray  # Shape: (T, H, W, C), uint8
    step: int
    episode_reward: float
    episode_length: int


@dataclass
class TrainResult:
    """Result of train_ppo containing final state and summary."""

    training_state: TrainingState
    final_metrics: Dict[str, Any]
    eval_history: List[Dict[str, Any]]
    total_steps: int
    total_iterations: int
