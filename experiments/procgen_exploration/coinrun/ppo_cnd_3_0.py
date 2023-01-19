import time
import torch

import RLAgents

import models.ppo_cnd_3_0.src.model_ppo        as ModelPPO
import models.ppo_cnd_3_0.src.model_cnd_target as ModelCNDTarget
import models.ppo_cnd_3_0.src.model_cnd        as ModelCND
import models.ppo_cnd_3_0.src.config           as Config

#torch.cuda.set_device("cuda:1")
  
path = "models/ppo_cnd_3_0/" 

config  = Config.Config()   

#config.envs_count = 1

envs = RLAgents.MultiEnvSeq("procgen-coinrun-v0", RLAgents.WrapperProcgenExploration, config.envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-coinrun-v0", RLAgents.WrapperProcgenExplorationRender, config.envs_count)

agent = RLAgents.AgentPPOCND(envs, ModelPPO, ModelCNDTarget, ModelCND, config)


max_iterations = 500000


trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 128)
trainig.run() 

'''
agent.load(path)
agent.disable_training()

episodes = 0
total_score = 0.0
reward_sum = 0.0

while True:
    reward, done, info = agent.main()

    #envs.render(0)
    #agent.render(0)
    
    reward_sum+= reward

    if done:
        episodes+= 1
        total_score+= reward_sum
        reward_sum = 0
        print("DONE ", episodes, total_score/episodes)
'''