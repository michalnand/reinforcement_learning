import time
import RLAgents

import models.ppo_baseline.src.model            as Model
import models.ppo_baseline.src.config           as Config

path = "models/ppo_baseline/"

config  = Config.Config()

config.envs_count = 1

#envs = RLAgents.MultiEnvParallelOptimised("BreakoutNoFrameskip-v4", RLAgents.WrapperAtari, config.envs_count)
envs = RLAgents.MultiEnvSeq("BreakoutNoFrameskip-v4", RLAgents.WrapperAtari, config.envs_count)

agent = RLAgents.AgentPPO(envs, Model, config)

'''
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
