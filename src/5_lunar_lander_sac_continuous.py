import gym
import numpy
import agents.sac_simple

import models.lunar_lander_sac_simple.src.model
import models.lunar_lander_sac_simple.src.config

import time



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

        return obs, reward, done, info


env = gym.make("LunarLanderContinuous-v2")
env = SetRewardRange(env)



model  = models.lunar_lander_sac_simple.src.model
#config = models.lunar_lander_sac_simple.src.config.Config()
 
agent = agents.sac_simple.Agent(env, model)


while agent.iterations < 100000:
    agent.main(enabled_training=True)

    if agent.iterations%128 == 0:
        env.render()
        print(agent.iterations, agent.score)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main(enabled_training=False)
    env.render()
    time.sleep(0.01)
