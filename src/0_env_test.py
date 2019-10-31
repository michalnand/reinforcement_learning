import gym
import numpy

import common.atari_wrapper

#env = gym.make("MountainCar-v0")
#env = gym.make("Pong-v4")
#env = gym.make("Breakout-v4")
#env = gym.make("SpaceInvaders-v4")
#env = gym.make("MsPacman-v0")

env = common.atari_wrapper.Create(env)

env.reset()

obs             = env.observation_space
actions_count   = env.action_space.n


print(obs, actions_count)


while True:
    action = numpy.random.randint(actions_count)
    observation, reward, done, info = env.step(action)
    env.render()

    if reward != 0:
        print(reward)
    if done:
        env.reset()