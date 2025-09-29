import jax
import jax.numpy as jp
import mujoco_playground

class EpisodeWrapper:
    def __init__(self, env: mujoco_playground.MjxEnv, max_len: int):
        self.env = env
        self.max_len = max_len

    def step(self, state, action):
        next_state = self.env.step(state, action)
        next_state.info["step_counter"] += 1
        truncated = jp.logical_or(next_state.info["truncated"],
                                  next_state.info["step_counter"] >= self.max_len)
        next_state.info["truncated"] = truncated
        next_state = next_state.replace(done=jp.logical_or(next_state.done, truncated).astype(float))
        return next_state
    
    def reset(self, rng):
        next_state = self.env.reset(rng)
        next_state.info["step_counter"] = jax.random.randint(rng, (), 0, 1000)
        next_state.info["truncated"] = False
        return next_state
    
    @property
    def observation_size(self):
        return self.env.observation_size
    
    @property
    def action_size(self):
        return self.env.action_size