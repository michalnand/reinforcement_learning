import gym
import common.atari_wrapper
import agents.dqn

import numpy
import time

#env = common.env_atari.Create("BreakoutDeterministic-v4", (80, 80))
#env = common.env_atari.Create("Enduro-v0")
#env = common.env_atari.Create("MsPacman-v0")
#env = common.env_atari.Create("PongDeterministic-v0", (64, 64)) 
#env = common.env_atari.Create("SpaceInvaders-v0")
#env = common.env_atari.Create("Seaquest-v0")


env = gym.make("Pong-v4") 
env = common.atari_wrapper.Create(env, 96, 96, 4) 

env.reset()


#common.atari_wrapper.show_state(observation)
import models.generic_net.src.model
import models.generic_net.src.config

model  = models.generic_net.src.model
config = models.generic_net.src.config.Config()

agent = agents.dqn.Agent(env, model, config)

save_path = "./models/generic_net/"



while agent.iterations < 10000000:
    agent.main()

    if agent.iterations%1024 == 0:
        agent._print()
        #env.render()

    if agent.iterations%100000 == 0:
        agent.save(save_path)

agent.save(save_path)


print("training done")

'''
agent.load(save_path)
agent.disable_training()

import time

while True:
    agent.main()
    env.render()
    time.sleep(0.025)
'''