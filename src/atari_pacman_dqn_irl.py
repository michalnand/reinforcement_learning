import gym
import common.atari_wrapper
import agents.dqn_imitation

import numpy
import time

import models.pacman_irl.src.model
import models.pacman_irl.src.config

import models.atari_dqn.pacman.src.model as ExpertModel


model  = models.pacman_irl.src.model
config = models.pacman_irl.src.config.Config()

save_path = "./models/pacman_irl/"

env = gym.make("MsPacmanNoFrameskip-v4") 
env = common.atari_wrapper.Create(env, 96, 96, 4) 

env.reset()

print("creating expert model")
expert_model = ExpertModel.Model(env.observation_space.shape, env.action_space.n)
print("loading expert model")
expert_model.load("./models/atari_dqn/pacman/")

agent = agents.dqn_imitation.Agent(env, model, config, expert_model, save_path)


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
