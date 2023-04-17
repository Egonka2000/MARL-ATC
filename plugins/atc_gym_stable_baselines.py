from stable_baselines3 import PPO, A2C
from plugins.atc_gym_env import AtcGymEnv
from plugins.atc_gym_callback import AtcCallback
import bluesky as bs
import os
import supersuit as ss
from stable_baselines3.common import on_policy_algorithm

class Agent():
    def __init__(self, train_mode, agents):
        self.agents = agents
        self.train_mode = train_mode
        self.train_started = False
        self.env = ss.pettingzoo_env_to_vec_env_v1(AtcGymEnv(self.train_mode))
        self.env = ss.concat_vec_envs_v1(self.env, 1, base_class='stable_baselines3')
        #if not self.train_mode:
        #self.env = AtcGymEnv(train_mode)
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
            if self.train_mode:
                raise Exception("Could not load model, so build the model for training...")
            else:
                self.model = PPO.load("{}/PPO_Model.h1".format(self.models_dir))
                print("Successfully loaded model")
        except:
            self.model = PPO(
                "MultiInputPolicy",
                env=self.env,
                verbose=0,
                tensorboard_log=self.logdir,
                )
            #self.env = AtcGymEnv(self.train_mode)
            #self.callback = AtcCallback(env=self.env)

    def update_plugin(self):
        if not self.train_mode:
            if self.done:
                self.obs = self.env.reset()
                self.done = False
                return
            #self.obs = self.env.reset() #.get_agents_and_nearest_ac_intruders_states(self.env.get_distance_matrix_ac())
            actions = {}
            for agent in self.agents:
                action, _states = self.model.predict(self.obs[agent])
                actions[agent] = action
            self.obs, reward, self.done, info = self.env.step(actions)
            #self.total_reward += reward
            print(reward)
            #print('Total Reward: {}'.format(self.total_reward))
        else:
            if not self.train_started:
                self.train()

    def train(self):
        print("Train!!!")
        self.train_started = True
        self.model.learn(total_timesteps=2000000, tb_log_name="PPO")
        print("Done")  
        self.model.save("{}/PPO_Model.h4".format(self.models_dir))
        bs.sim.quit()