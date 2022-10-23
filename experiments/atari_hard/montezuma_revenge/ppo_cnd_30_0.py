import time
import torch
import numpy

import RLAgents

import models.ppo_cnd_30_0.src.model_ppo        as ModelPPO
import models.ppo_cnd_30_0.src.model_cnd_target as ModelCNDTarget
import models.ppo_cnd_30_0.src.model_cnd        as ModelCND
import models.ppo_cnd_30_0.src.config           as Config

#torch.cuda.set_device("cuda:0")
  
path = "models/ppo_cnd_30_0/"

config  = Config.Config() 

#config.envs_count = 1
 
envs = RLAgents.MultiEnvParallelOptimised("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperMontezuma, config.envs_count)
#envs = RLAgents.MultiEnvSeq("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperMontezuma, config.envs_count)
#envs = RLAgents.MultiEnvSeq("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperMontezumaVideo, config.envs_count)
 
agent = RLAgents.AgentPPOCND(envs, ModelPPO, ModelCNDTarget, ModelCND, config)
 
max_iterations = 1000000 
 
 
trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 128)
trainig.run() 

'''
agent.load(path)
agent.disable_training()


while True:
    reward, done, info = agent.main()

    envs.render(0)
    #time.sleep(0.01)
'''