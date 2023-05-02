from stable_baselines3 import PPO, A2C
from plugins.atc_gym_env import AtcGymEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import bluesky as bs
import os
import supersuit as ss
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

class Agent():
    def __init__(self, train_mode, agents):
        self.agents = agents
        self.train_mode = train_mode
        self.train_started = False
        self.env = ss.pettingzoo_env_to_vec_env_v1(AtcGymEnv(self.train_mode))
        #self.env = MarkovVectorEnv(AtcGymEnv(self.train_mode))
        self.env = ss.concat_vec_envs_v1(self.env, 1, base_class='stable_baselines3')
        self.env = VecMonitor(self.env)
        self.total_reward = 0
        self.done = True

        self.logdir = f"logs"
        self.models_dir = f"models"

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        self.buildModel()
        
    def buildModel(self):
        try:
            if self.train_mode:
                raise Exception("Could not load model, so build the model for training...")
            else:
                self.model = PPO.load("{}/PPO_16_400000".format(self.models_dir), env=self.env)
                print("Successfully loaded model")
        except:
            self.model = PPO(
                "MultiInputPolicy",
                env=self.env,
                verbose=0,
                #n_steps=256,
                ent_coef=0.001,
                tensorboard_log=self.logdir,
                )

    def update_plugin(self):
        if not self.train_mode:
            if self.done:
                self.obs = self.env.reset()
                self.done = False
                return
            action, _states = self.model.predict(self.obs)
            self.obs, rewards, term, info = self.env.step(action)
            for rew in rewards:
                self.total_reward += rew
            if any(term):
                print("Episode total reward: {}".format(self.total_reward))
                self.total_reward = 0
        else:
            if not self.train_started:
                self.train()

    def train(self):
        print("Train!!!")
        self.train_started = True
        TIMESTEPS = 200000
        for i in range(1,20):
            self.model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_16")
            self.model.save("{}/PPO_16_{}".format(self.models_dir, TIMESTEPS*i))
        print("Done")
        bs.sim.quit()