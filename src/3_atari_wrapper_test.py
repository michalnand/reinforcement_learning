import gym
import numpy
import time

import common.atari_wrapper

#env = gym.make("Pong-v4")
env = gym.make("Breakout-v4")
#env = gym.make("SpaceInvaders-v4")
#env = gym.make("MsPacman-v4")
#env = gym.make("Seaquest-v4") 
#env = gym.make("Qbert-v4") 


env = common.atari_wrapper.Create(env)

env.reset()

obs             = env.observation_space
actions_count   = env.action_space.n


print(obs, actions_count)


while True:
    action = numpy.random.randint(actions_count)
    observation, reward, done, info = env.step(action)
    env.render()

    #common.atari_wrapper.observation_show(observation)    

    if reward != 0:
        print("reward = ", reward)

    if done:
        print("\n\nGAME DONE\n\n")
        env.reset()
        time.sleep(0.5)


    time.sleep(0.01)
