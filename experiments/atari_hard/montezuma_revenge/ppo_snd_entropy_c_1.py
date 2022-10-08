import time
import torch

import RLAgents

import models.ppo_snd_entropy_c_1.src.model_ppo         as ModelPPO
import models.ppo_snd_entropy_c_1.src.model_snd_target  as ModelSNDTarget
import models.ppo_snd_entropy_c_1.src.model_snd         as ModelSND
import models.ppo_snd_entropy_c_1.src.model_entropy     as ModelEntropy
import models.ppo_snd_entropy_c_1.src.config            as Config

#torch.cuda.set_device("cuda:0")
 
path = "models/ppo_snd_entropy_c_1/"

config  = Config.Config() 

config.envs_count = 1
 
#envs = RLAgents.MultiEnvParallelOptimised("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperMontezuma, config.envs_count)
envs = RLAgents.MultiEnvSeq("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperMontezuma, config.envs_count)
#envs = RLAgents.MultiEnvSeq("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperMontezumaVideo, config.envs_count)

agent = RLAgents.AgentPPOSNDEntropy(envs, ModelPPO, ModelSNDTarget, ModelSND, ModelEntropy, config)

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
    time.sleep(0.01)
