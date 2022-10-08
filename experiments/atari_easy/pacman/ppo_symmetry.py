import time

import RLAgents

import models.ppo_symmetry.src.model            as Model
import models.ppo_symmetry.src.config           as Config

path = "models/ppo_symmetry/"

config  = Config.Config()

#config.envs_count = 1

envs = RLAgents.MultiEnvParallelOptimised("MsPacmanNoFrameskip-v4", RLAgents.WrapperAtari, config.envs_count)
#envs = RLAgents.MultiEnvSeq("MsPacmanNoFrameskip-v4", RLAgents.WrapperAtari, config.envs_count)

agent = RLAgents.AgentPPOSymmetry(envs, Model, config)


max_iterations = 1000000
trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 128)
trainig.run() 

'''
agent.load(path) 
agent.disable_training()
while True:
    reward, done, _ = agent.main()

    envs.render(0)
    #agent.render(0)
    time.sleep(0.01)
'''