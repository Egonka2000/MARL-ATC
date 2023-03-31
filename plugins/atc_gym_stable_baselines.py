from stable_baselines3 import PPO, A2C
from plugins.atc_gym_env import AtcGymEnv
from plugins.atc_gym_callback import AtcCallback
import bluesky as bs
import os

class Agent():
    def __init__(self, train_mode):
        self.env = AtcGymEnv(train_mode)
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
        #Legjobb modell A2C_Model.h7, A2C_Model.h10 is nagyon jó
        try:
            if self.env.train_mode:
                raise Exception("Could not load model, so build the model for training...")
            else:
                self.model = A2C.load("{}/A2C_Model.h10".format(self.models_dir))
                print("Successfully loaded model")
        except:
            self.model = A2C(
                "MultiInputPolicy", 
                self.env, 
                verbose=0,
                tensorboard_log=self.logdir,
            )
            self.callback = AtcCallback(env=self.env)

    def update_plugin(self):
        if not self.env.train_mode:
            if self.done:
                self.obs = self.env.reset()
                self.done = False
                return
            obs = self.env.get_agent_and_nearest_ac_intruders_states(self.env.get_distance_matrix_ac(), self.env.current_agent_idx)
            action, _states = self.model.predict(obs)
            obs, reward, self.done, info = self.env.step(action)
            self.total_reward += reward
            print('Total Reward: {}'.format(self.total_reward))
        else:
            if not self.env.train_started:
                self.train()

    def train(self):
        print("Train!!!")
        self.env.train_started = True
        self.model.learn(total_timesteps=1000000, tb_log_name="A2C", callback=self.callback)
        print("Done")  
        self.model.save("{}/A2C_Model.h11".format(self.models_dir))
        bs.sim.quit()