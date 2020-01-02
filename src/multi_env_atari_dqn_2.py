import gym
import common.atari_wrapper
import common.env_multi


import agents.dqn


import numpy
import time

import models.multi_agent.atari_dqn_2.src.model
import models.multi_agent.atari_dqn_2.src.config


model  = models.multi_agent.atari_dqn_2.src.model
config = models.multi_agent.atari_dqn_2.src.config.Config()

save_path = "./models/multi_agent/atari_dqn_2/" 

env = gym.make("MsPacmanNoFrameskip-v4") 
env = common.atari_wrapper.Create(env, 96, 96, 4) 

width  = 96
height = 96
frame_stacking = 4

envs = []

envs.append(common.atari_wrapper.Create(gym.make("BreakoutNoFrameskip-v4"), width, height, frame_stacking))
envs.append(common.atari_wrapper.Create(gym.make("MsPacmanNoFrameskip-v4"), width, height, frame_stacking))
envs.append(common.atari_wrapper.Create(gym.make("SeaquestNoFrameskip-v4"), width, height, frame_stacking))
envs.append(common.atari_wrapper.Create(gym.make("QbertNoFrameskip-v4"), width, height, frame_stacking))


env = common.env_multi.EnvMulti(envs, 8192)
env.reset()




agent = agents.dqn.Agent(env, model, config, save_path)

score_best = -10000.0
while agent.iterations < 40000000:
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
