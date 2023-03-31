from stable_baselines3.common.callbacks import BaseCallback
from bluesky import traf

class AtcCallback(BaseCallback):

    def __init__(self, env, verbose=0):
        super(AtcCallback, self).__init__(verbose)
        self.env = env
        self._idxs = list(range(0, self.env.max_ac))
        self.iterator_list = iter(self._idxs)
        self.default = self._idxs[0]
    
    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_rollout_end(self) -> None:
        if(self.env.current_agent_idx < self.env.max_ac - 1):
            self.env.current_agent_idx = self.env.current_agent_idx + 1
        else:
            self.env.current_agent_idx = 0