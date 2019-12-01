import gym
import numpy
import agents.a2c

import models.mountain_car_a2c.src.model
import models.mountain_car_a2c.src.config

import time


gym.envs.register(
    id='MountainCarCustom-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=4096      # MountainCar-v0 uses 200
)

env = gym.make("MountainCarCustom-v0")


class SetRewardRange(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward < 0:
            reward = -0.001

        if done: 
            reward = 1.0
        
        return obs, reward, [done, done], info


env = SetRewardRange(env)
env.reset()

obs             = env.observation_space
actions_count   = env.action_space.n




model  = models.mountain_car_a2c.src.model
config = models.mountain_car_a2c.src.config.Config()

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
