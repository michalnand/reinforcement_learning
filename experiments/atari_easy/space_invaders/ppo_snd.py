import gym
import numpy
import time
import torch

import RLAgents

import models.ppo_snd.src.model_ppo        as ModelPPO
import models.ppo_snd.src.model_snd_target as ModelSNDTarget
import models.ppo_snd.src.model_snd        as ModelSND
import models.ppo_snd.src.config           as Config

path = "models/ppo_snd/" 

config  = Config.Config()
#config.envs_count = 1

envs = RLAgents.MultiEnvParallelOptimised("SpaceInvadersNoFrameskip-v4", RLAgents.WrapperAtariNoRewards, config.envs_count)
#envs = RLAgents.MultiEnvSeq("SpaceInvadersNoFrameskip-v4", RLAgents.WrapperAtariNoRewards, config.envs_count)

agent = RLAgents.AgentPPOSND(envs, ModelPPO, ModelSNDTarget, ModelSND, config)

max_iterations = 1000000
trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 128)
trainig.run() 

 
'''
agent.load(path)
agent.disable_training()
while True:
    reward, done, _ = agent.main()

    envs.render(0)
    time.sleep(0.01)
'''