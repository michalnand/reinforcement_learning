import gym_super_mario_bros
import common.super_mario_wrapper

import agents.a2c

import numpy
import time

import models.super_mario.a2c.src.model
import models.super_mario.a2c.src.config





save_path = "./models/super_mario/a2c/"
paralel_envs_count = 8

envs = [] 

for i in range(paralel_envs_count):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    #env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')
    env = common.super_mario_wrapper.Create(env)
    env.reset()
    env.seed(i)

    envs.append(env)

obs             = envs[0].observation_space
actions_count   = envs[0].action_space.n

model  = models.super_mario.a2c.src.model
config = models.super_mario.a2c.src.config.Config()
 
agent = agents.a2c.Agent(envs, model, config, save_path)



score_best = -10000.0
while agent.iterations < 10000000:
    agent.main()    
    if agent.iterations%65536 == 0:
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
