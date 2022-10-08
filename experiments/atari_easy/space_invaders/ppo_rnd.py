import gym
import numpy
import time
import torch

import RLAgents

import models.ppo_rnd.src.model_ppo   as ModelPPO
import models.ppo_rnd.src.model_rnd   as ModelRND
import models.ppo_rnd.src.config      as Config

path = "models/ppo_rnd/"

config  = Config.Config()
#config.envs_count = 1

envs = RLAgents.MultiEnvParallelOptimised("SpaceInvadersNoFrameskip-v4", RLAgents.WrapperAtariNoRewards, config.envs_count)
#envs = RLAgents.MultiEnvSeq("SpaceInvadersNoFrameskip-v4", RLAgents.WrapperAtariNoRewards, config.envs_count)

agent = RLAgents.AgentPPORND(envs, ModelPPO, ModelRND, config)

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