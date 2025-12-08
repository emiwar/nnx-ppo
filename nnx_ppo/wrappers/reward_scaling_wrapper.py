import mujoco_playground

class RewardScalingWrapper:

    def __init__(self, env: mujoco_playground.MjxEnv, reward_scale: float) -> None:
        self.env = env
        self.reward_scale = reward_scale

    def reset(self, *args):
        next_state = self.env.reset(*args)
        next_state = next_state.replace(reward = self.reward_scale * next_state.reward)
        return next_state
    
    def step(self, *args):
        next_state = self.env.step(*args)
        next_state = next_state.replace(reward = self.reward_scale * next_state.reward)
        return next_state
    
    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def action_size(self):
        return self.env.action_size
