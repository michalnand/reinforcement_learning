import time
import numpy
import RLAgents


envs_count = 128
 
envs = RLAgents.MultiEnvParallel("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperMontezuma, envs_count)
#envs = RLAgents.MultiEnvSeq("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperMontezuma, envs_count)

for i in range(envs_count):
    envs.reset(i)


actions_count   = envs.action_space.n
k               = 0.02
fps             = 0.0
while True:

    time_start = time.time()
    actions = numpy.random.randint(0, actions_count, envs_count)
    states, rewards, dones, infos = envs.step(actions)
    time_stop  = time.time()

    fps = (1.0 - k)*fps + k*envs_count/(time_stop - time_start)

    for i in range(envs_count):
        if dones[i]:
            envs.reset(i)

    print("fps = ", round(fps, 2))

