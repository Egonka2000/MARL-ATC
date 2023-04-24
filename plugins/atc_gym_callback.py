from stable_baselines3.common.callbacks import BaseCallback

class AtcCallback(BaseCallback):

    def __init__(self, env, verbose=0):
        super(AtcCallback, self).__init__(verbose)
        self.env1 = env
        self.rollout_ends = 0


    def _on_step(self) -> bool:
        return super()._on_step()
        
    def _on_rollout_end(self) -> None:
        if(self.env1.all_rewards != 0):
            print(self.env1.all_rewards)
        self.reward_mean = self.env1.all_rewards / self.env1.max_ac
        self.logger.record("episode_reward_mean", self.reward_mean)