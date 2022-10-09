import time
import numpy

import RLAgents

envs_count = 4
  


#envs = RLAgents.MultiEnvSeq("procgen-coinrun-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-starpilot-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-caveflyer-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-dodgeball-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-fruitbot-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-chaser-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-miner-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-jumper-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-leaper-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-maze-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-bigfish-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-heist-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-climber-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-plunder-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-ninja-v0", RLAgents.WrapperProcgenEasyRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-bossfight-v0", RLAgents.WrapperProcgenEasyRender, envs_count)


#envs = RLAgents.MultiEnvSeq("procgen-coinrun-v0", RLAgents.WrapperProcgenExplorationRender, envs_count)
envs = RLAgents.MultiEnvSeq("procgen-caveflyer-v0", RLAgents.WrapperProcgenExplorationRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-leaper-v0", RLAgents.WrapperProcgenExplorationRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-jumper-v0", RLAgents.WrapperProcgenExplorationRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-maze-v0", RLAgents.WrapperProcgenExplorationRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-heist-v0", RLAgents.WrapperProcgenExplorationRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-climber-v0", RLAgents.WrapperProcgenExplorationRender, envs_count)
#envs = RLAgents.MultiEnvSeq("procgen-ninja-v0", RLAgents.WrapperProcgenExplorationRender, envs_count)
 
for i in range(envs_count):
    envs.reset(i)

fps = 0.0
k   = 0.1

while True:

    t_start = time.time()
    actions = numpy.random.randint(0, envs.action_space.n, size=envs_count)
    obs, rewards, dones, infos = envs.step(actions)
    t_stop = time.time()

    fps = (1.0 - k)*fps + k*envs_count*1.0/(t_stop - t_start)
    
    for i in range(envs_count):
        if dones[i]:
            envs.reset(i)

    print("fps = ", fps, rewards)
