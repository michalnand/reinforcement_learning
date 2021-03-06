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

score_best = -10000.0
while agent.iterations < 10000000:
    agent.main()    
    if agent.iterations%100000 == 0:
        if agent.training_stats.game_score_smooth > score_best:
            score_best = agent.training_stats.game_score_smooth
            agent.save()
            
            print("\n\n\n")
            print("saving best agent")
            print("iteration = ", agent.iterations)
            print("score_best = ", score_best)
            print("\n\n\n")

print("training done")

agent.load()
agent.disable_training()

agent.iterations = 0
while agent.iterations  < 1000000:
    agent.main()

print("testing done")
