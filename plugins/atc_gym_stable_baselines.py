from stable_baselines3 import PPO, A2C
from plugins.atc_gym_env import AtcGymEnv
from plugins.atc_gym_callback import AtcCallback
import bluesky as bs
import os
import supersuit as ss
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import ray
from ray import air, tune
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole

class Agent():
    def __init__(self, train_mode):
        self.env = AtcGymEnv(train_mode)
        register_env('atc-gym-env', lambda cfg:AtcGymEnv(train_mode, cfg))
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
                self.model = PPO.load("{}/PPO_Model.h1".format(self.models_dir))
                print("Successfully loaded model")
        except:
            #self.config = PPOConfig().training(gamma=0.9, lr=0.01, kl_coeff=0.3).resources(num_gpus=0).rollouts(num_rollout_workers=4).environment(env='atc-gym-env', env_config={"num_workers":3}, observation_space=self.env.observation_space, action_space=self.env.action_space)
            # Build a Algorithm object from the config and run 1 training iteration.
            #self.callback = AtcCallback(env=self.env)
            pass

    def update_plugin(self):
        if not self.env.train_mode:
            if self.done:
                self.obs = self.env.reset()
                self.done = False
                return
            obs = self.env.get_agents_and_nearest_ac_intruders_states(self.env.get_distance_matrix_ac())
            while True:
                obs, rew, done, info = self.env.step(
                    {'agent-0': self.env.action_space.sample(), 'agent-1': self.env.action_space.sample(), 'agent-2': self.env.action_space.sample()}
                )
            actions = {}
            for agent in self.env.agents:
                action, _states = self.model.predict(obs[agent])
                actions[agent] = action
            obs, reward, self.done, info = self.env.step(actions)
            #self.total_reward += reward
            print(reward)
            #print('Total Reward: {}'.format(self.total_reward))
        else:
            if not self.env.train_started:
                self.train()

    def train(self):
        print("Train!!!")
        self.env.train_started = True
        if ray.is_initialized(): ray.shutdown()
        ray.init() #Prints the dashboard running on a local port
        tune.run('PPO', config={"env": 'atc-gym-env', 'observation_space': self.env.observation_space, 'action_space': self.env.action_space}, stop={"timesteps_total": 10 })
        ray.shutdown()
        #self.model.train()
        #self.model.learn(total_timesteps=2000000, tb_log_name="PPO")
        print("Done")  
        #self.model.save("{}/PPO_Model.h4".format(self.models_dir))
        bs.sim.quit()