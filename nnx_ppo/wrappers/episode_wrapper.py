import jax
import jax.numpy as jp

from nnx_ppo.algorithms.types import RLEnv, EnvState


class EpisodeWrapper:
    def __init__(self, env: RLEnv, max_len: int):
        self.env = env
        self.max_len = max_len

    def step(self, state: EnvState, action) -> EnvState:
        next_state = self.env.step(state, action)
        next_state.info["step_counter"] = state.info["step_counter"] + 1
        truncated = jp.logical_or(
            next_state.info.get("truncated", False),
            next_state.info["step_counter"] >= self.max_len,
        )
        next_state.info["truncated"] = truncated
        next_state = next_state.replace(  # type: ignore[attr-defined]
            done=jp.logical_or(next_state.done, truncated).astype(float)
        )
        return next_state

    def reset(self, rng) -> EnvState:
        base_rng, step_counter_rng = jax.random.split(rng)
        next_state = self.env.reset(base_rng)
        next_state.info["step_counter"] = jax.random.randint(
            step_counter_rng, (), 0, self.max_len // 2
        )
        next_state.info["truncated"] = False
        return next_state

    @property
    def observation_size(self):
        return self.env.observation_size  # type: ignore[attr-defined]

    @property
    def action_size(self):
        return self.env.action_size  # type: ignore[attr-defined]