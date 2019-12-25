import gym_super_mario_bros
import common.doom_wrapper

import agents.dqn

import numpy
import time

import models.doom.src.model
import models.doom.src.config


model  = models.doom.src.model
config = models.doom.src.config.Config()

save_path = "./models/doom/"

env = common.doom_wrapper.Create("defend_the_line", 96, 96, 4)

env.reset()



agent = agents.dqn.Agent(env, model, config, save_path)

agent.load()
agent.disable_training()

while True:
    agent.main()
    env.render()

print("testing done")
