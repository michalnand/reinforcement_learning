import RLAgents

import models.ppo_cnd_10_0.src.model_ppo        as ModelPPO_10_0
import models.ppo_cnd_10_0.src.model_cnd_target as ModelCNDTarget_10_0
import models.ppo_cnd_10_0.src.model_cnd        as ModelCND_10_0
import models.ppo_cnd_10_0.src.config           as Config_10_0

import models.ppo_cnd_10_1.src.model_ppo        as ModelPPO_10_1
import models.ppo_cnd_10_1.src.model_cnd_target as ModelCNDTarget_10_1
import models.ppo_cnd_10_1.src.model_cnd        as ModelCND_10_1
import models.ppo_cnd_10_1.src.config           as Config_10_1

import models.ppo_cnd_10_2.src.model_ppo        as ModelPPO_10_2
import models.ppo_cnd_10_2.src.model_cnd_target as ModelCNDTarget_10_2
import models.ppo_cnd_10_2.src.model_cnd        as ModelCND_10_2
import models.ppo_cnd_10_2.src.config           as Config_10_2


config_10_0      = Config_10_0.Config() 
envs_10_0        = RLAgents.MultiEnvSeq("procgen-climber-v0", RLAgents.WrapperProcgenExploration, config_10_0.envs_count)
agent_10_0       = RLAgents.AgentPPOCND(envs_10_0, ModelPPO_10_0, ModelCNDTarget_10_0, ModelCND_10_0, config_10_0)
saving_path_10_0 = "models/ppo_cnd_10_0/"

config_10_1      = Config_10_1.Config() 
envs_10_1        = RLAgents.MultiEnvSeq("procgen-climber-v0", RLAgents.WrapperProcgenExploration, config_10_1.envs_count)
agent_10_1       = RLAgents.AgentPPOCND(envs_10_1, ModelPPO_10_1, ModelCNDTarget_10_1, ModelCND_10_1, config_10_1)
saving_path_10_1 = "models/ppo_cnd_10_1/"

config_10_2      = Config_10_2.Config() 
envs_10_2        = RLAgents.MultiEnvSeq("procgen-climber-v0", RLAgents.WrapperProcgenExploration, config_10_2.envs_count)
agent_10_2       = RLAgents.AgentPPOCND(envs_10_2, ModelPPO_10_2, ModelCNDTarget_10_2, ModelCND_10_2, config_10_2)
saving_path_10_2 = "models/ppo_cnd_10_2/"



 
envs = []
envs.append(config_10_0)
envs.append(config_10_1)
envs.append(config_10_2)


agents = [] 
agents.append(agent_10_0)
agents.append(agent_10_1)
agents.append(agent_10_2)

saving_paths = []
saving_paths.append(saving_path_10_0)
saving_paths.append(saving_path_10_1)
saving_paths.append(saving_path_10_2)



max_iterations = 500000

training = RLAgents.TrainingIterationsMultiRuns(envs, agents, max_iterations, saving_paths, 128)
training.run()
