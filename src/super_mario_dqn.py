import gym_super_mario_bros
import common.super_mario_wrapper

import agents.dqn

import numpy
import time

import models.super_mario.dqn.src.model
import models.super_mario.dqn.src.config


model  = models.super_mario.dqn.src.model
config = models.super_mario.dqn.src.config.Config()

save_path = "./models/super_mario/dqn/"

#env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')
env = common.super_mario_wrapper.Create(env)

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
