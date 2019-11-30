import gym
import common.atari_wrapper
import agents.dqn

import numpy
import time

import models.atari_qbert_dqn.src.model
import models.atari_qbert_dqn.src.config


model  = models.atari_qbert_dqn.src.model
config = models.atari_qbert_dqn.src.config.Config()

save_path = "./models/atari_qbert_dqn/"

env = gym.make("QbertNoFrameskip-v4") 
env = common.atari_wrapper.Create(env, 96, 96, 4) 

env.reset()

agent = agents.dqn.Agent(env, model, config, save_path)

while agent.iterations < 10000000:

    agent.main()    

    if agent.iterations%1000000 == 0:
        agent.save()


agent.save() 

print("training done")

agent.load()
agent.disable_training()

agent.iterations = 0
while agent.iterations  < 1000000:
    agent.main()

print("testing done")
