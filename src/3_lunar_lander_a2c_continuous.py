import gym
import numpy
import agents.a2c_continuous

import models.lunar_lander_a2c_continuous.src.model
import models.lunar_lander_a2c_continuous.src.config

import time


paralel_envs_count = 8

class SetRewardRange(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        reward = reward / 10.0

        if reward < -1.0:
            reward = -1.0

        if reward > 1.0: 
            reward = 1.0

        return obs, reward, [done, done], info

envs = [] 

for i in range(paralel_envs_count):
    env = gym.make("LunarLanderContinuous-v2")

    env = SetRewardRange(env)
    obs = env.reset()
    env.seed(i)

    envs.append(env)

obs             = envs[0].observation_space
actions_size    = envs[0].action_space.shape[0]


model  = models.lunar_lander_a2c_continuous.src.model
config = models.lunar_lander_a2c_continuous.src.config.Config()
 
agent = agents.a2c_continuous.Agent(envs, model, config)


while agent.iterations < 100000:
    agent.main()

    if agent.iterations%128 == 0:
        envs[0].render()
        print(agent.iterations, agent.score)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    envs[0].render()
    time.sleep(0.01)
