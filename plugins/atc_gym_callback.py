from stable_baselines3.common.callbacks import BaseCallback

class AtcCallback(BaseCallback):

    def __init__(self, env, verbose=0):
        super(AtcCallback, self).__init__(verbose)
        self.env = env
        self.rollout_ends = 0

    def _on_step(self) -> bool:
        return super()._on_step()
        
    def _on_rollout_end(self) -> None:
        self.rollout_ends += 1
        self.env.current_agent_idx = self.rollout_ends % self.env.max_ac