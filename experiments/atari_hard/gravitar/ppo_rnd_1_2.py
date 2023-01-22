import time

import RLAgents

import models.ppo_rnd_1_2.src.model_ppo               as ModelPPO
import models.ppo_rnd_1_2.src.model_rnd               as ModelRND
import models.ppo_rnd_1_2.src.config                  as Config

 
path = "models/ppo_rnd_1_2/" 
 
config  = Config.Config() 

#config.envs_count = 1 
 
envs = RLAgents.MultiEnvParallelOptimised("GravitarNoFrameskip-v4", RLAgents.WrapperMontezuma, config.envs_count)
#envs = RLAgents.MultiEnvSeq("GravitarNoFrameskip-v4", RLAgents.WrapperMontezuma, config.envs_count)
 
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
    #time.sleep(0.01)
    agent.render(0)
'''