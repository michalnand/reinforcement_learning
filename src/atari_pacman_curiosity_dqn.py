import gym
import common.atari_wrapper
import agents.dqn_curiosity

import numpy
import time

import models.atari_dqn.pacman_curiosity_dqn.src.icm_model
import models.atari_dqn.pacman_curiosity_dqn.src.config


model  = models.atari_dqn.pacman_curiosity_dqn.src.icm_model
config = models.atari_dqn.pacman_curiosity_dqn.src.config.Config()

save_path = "./models/atari_dqn/pacman_curiosity_dqn/" 

#env = gym.make("MsPacmanNoFrameskip-v4") 
env = gym.make("PongNoFrameskip-v4") 
env = common.atari_wrapper.Create(env, 96, 96, 4) 

env.reset()

agent = agents.dqn_curiosity.Agent(env, model, config, save_path)

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
