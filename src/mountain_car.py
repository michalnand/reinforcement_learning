import agents.dqn
import gym

import numpy

env = gym.make("MountainCar-v0")


obs = env.observation_space
actions = env.action_space.n
env.reset()


class EnvWrapper(gym.Wrapper):
    def __init__(self, env=None):
        super(EnvWrapper, self).__init__(env)
        self.actions_count   = env.action_space.n
        self.shape           = (1, 2)


env = EnvWrapper(env)


import models.fc_net.src.model
import models.fc_net.src.config

model  = models.fc_net.src.model
config = models.fc_net.src.config.Config()

agent = agents.dqn.Agent(env, model, config)

save_path = "./models/fc_net/"
 
while agent.iterations < 1000000:
    agent.main()

    if agent.iterations%64 == 0:
        agent._print() 
        env.render()

    if agent.iterations%100000 == 0:
        agent.save(save_path)

agent.save(save_path)

print("program done")