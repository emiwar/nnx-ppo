"""Callback helpers for training logging."""
from typing import Callable

from nnx_ppo.algorithms.config import VideoData


def wandb_video_fn(
    key: str = "eval_video",
    fps: int = 30
) -> Callable[[VideoData], None]:
    """Create a video callback that logs to wandb.

    Args:
        key: The wandb log key for the video.
        fps: Frames per second for the video.

    Returns:
        A callback function compatible with train_ppo's video_fn parameter.

    Example:
        >>> result = train_ppo(
        ...     env, networks,
        ...     video_fn=wandb_video_fn(fps=50),
        ... )
    """
    import wandb
    def video_fn(data: VideoData) -> None:
        # Convert THWC to TCHW for wandb
        video_array = data.frames.transpose(0, 3, 1, 2)
        wandb.log(
            {key: wandb.Video(video_array, fps=fps, format="mp4")},
            step=data.step
        )
    return video_fn
