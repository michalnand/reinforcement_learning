import gym
import common.atari_wrapper
import agents.dqn

import numpy
import time

import models.atari_pacman_dqn.src.model
import models.atari_pacman_dqn.src.config


model  = models.atari_pacman_dqn.src.model
config = models.atari_pacman_dqn.src.config.Config()

save_path = "./models/atari_pacman_dqn/"

env = gym.make("MsPacman-v4") 
env = common.atari_wrapper.Create(env, 96, 96, 4) 

env.reset()

agent = agents.dqn.Agent(env, model, config, save_path)

while agent.iterations < 10000000:
    agent.main()

    if agent.iterations%1000 == 0:
        pass
        #env.render()

    if agent.iterations%100000 == 0:
        agent.save()

agent.save()


print("training done")

agent.load()
agent.disable_training()

while agent.iterations < 1000000:
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
