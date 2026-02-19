"""Mock environment for testing stateful networks with rollouts."""
import jax
import jax.numpy as jp
from flax import struct


@struct.dataclass
class MockEnvState:
    """Simple mock environment state."""
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    step_count: jax.Array
    info: dict
    metrics: dict = struct.field(default_factory=dict)

    def info_get(self, key, default):
        return default


class MockEnv:
    """A simple mock environment that resets after max_steps.

    This environment has:
    - obs_size dimensional observations (random noise)
    - action_size dimensional actions (ignored)
    - Resets when step_count >= max_steps
    """

    def __init__(self, obs_size: int, action_size: int, max_steps: int = 5):
        self.obs_size = obs_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.observation_size = obs_size

    def reset(self, key: jax.Array) -> MockEnvState:
        obs_key, = jax.random.split(key, 1)
        return MockEnvState(
            obs=jax.random.normal(obs_key, (self.obs_size,)),
            reward=jp.array(0.0),
            done=jp.array(False),
            step_count=jp.array(0),
            info={},
            metrics={},
        )

    def step(self, state: MockEnvState, action: jax.Array) -> MockEnvState:
        key = jax.random.PRNGKey(state.step_count + 1)
        new_obs = jax.random.normal(key, (self.obs_size,))
        new_step_count = state.step_count + 1
        done = new_step_count >= self.max_steps
        return MockEnvState(
            obs=new_obs,
            reward=jp.array(1.0),
            done=done,
            step_count=new_step_count,
            info={},
            metrics={},
        )
