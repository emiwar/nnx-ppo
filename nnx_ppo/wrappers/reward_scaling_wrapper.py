from typing import Any

from jaxtyping import PRNGKeyArray

from nnx_ppo.algorithms.types import EnvState, RLEnv


class RewardScalingWrapper:

    def __init__(self, env: RLEnv, reward_scale: float) -> None:
        self.env = env
        self.reward_scale = reward_scale

    def reset(self, rng: PRNGKeyArray) -> EnvState:
        next_state = self.env.reset(rng)
        return next_state.replace(reward=self.reward_scale * next_state.reward)  # type: ignore[attr-defined]

    def step(self, state: EnvState, action: Any) -> EnvState:
        next_state = self.env.step(state, action)
        return next_state.replace(reward=self.reward_scale * next_state.reward)  # type: ignore[attr-defined]

    @property
    def observation_size(self) -> Any:
        return self.env.observation_size  # type: ignore[attr-defined]

    @property
    def action_size(self) -> int:
        return self.env.action_size  # type: ignore[attr-defined]
