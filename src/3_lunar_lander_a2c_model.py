import gym
import numpy
import agents.a2c_model

import models.lunar_lander_a2c_model.src.model
import models.lunar_lander_a2c_model.src.config

import time


paralel_envs_count = 3

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
    env = gym.make("LunarLander-v2")

    env = SetRewardRange(env)
    env.reset()
    env.seed(i)

    envs.append(env)



obs             = envs[0].observation_space
actions_count   = envs[0].action_space.n




model  = models.lunar_lander_a2c_model.src.model
config = models.lunar_lander_a2c_model.src.config.Config()
 
agent = agents.a2c_model.Agent(envs, model, config)


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
