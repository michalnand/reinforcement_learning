import gym
import numpy
import agents.a2c

import models.lunar_lander_a2c.src.model
import models.lunar_lander_a2c.src.config

import time


env = gym.make("LunarLander-v2")


class SetRewardRange(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward < -1.0:
            reward = -1.0

        if reward > 1.0:
            reward = 1.0

        
        return obs, reward, [done, done], info
env = SetRewardRange(env)


env.reset()

obs             = env.observation_space
actions_count   = env.action_space.n




model  = models.lunar_lander_a2c.src.model
config = models.lunar_lander_a2c.src.config.Config()
 
agent = agents.a2c.Agent(env, model, config)


while agent.iterations < 1000000:
    agent.main()

    if agent.iterations%100 == 0:
        env.render()
        print(agent.iterations, agent.score)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
