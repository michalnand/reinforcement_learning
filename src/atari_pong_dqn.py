import gym
from gym import ObservationWrapper

import agents.dqn

import numpy
import time

import models.atari_pong_dqn.src.model
import models.atari_pong_dqn.src.config


model  = models.atari_pong_dqn.src.model
config = models.atari_pong_dqn.src.config.Config()

save_path = "./models/atari_pong_dqn/"

env = gym.make("Pong-v4")
env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=96)
env = gym.wrappers.FrameStack(env, 4)


class NumpyFrame(ObservationWrapper):
    def __init__(self, env):
        super(NumpyFrame, self).__init__(env)

    def _convert_observation(self, observation):
        return numpy.array(observation)/255.0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        result = self._convert_observation(observation)
        return result, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        result = self._convert_observation(observation)
        return result


env = NumpyFrame(env)


env.reset() 
 

agent = agents.dqn.Agent(env, model, config, save_path)

while agent.iterations < 10000000:
    agent.main()    

    if agent.iterations%100000 == 0:
        agent.save()

    if agent.iterations%1000 == 0:
        pass
        #env.render()

agent.save() 

print("training done")


agent.load()
agent.disable_training()

agent.iterations = 0
while agent.iterations  < 1000000:
    agent.main()

print("testing done")

'''

agent = agents.dqn.Agent(env, model, config, save_path, save_stats=False)
agent.load()
agent.disable_training()

while True:
    agent.main()
    env.render()
    time.sleep(0.1)
'''