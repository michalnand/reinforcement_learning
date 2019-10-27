import gym
import numpy
import agents.dqn

import models.fc_net.src.model
import models.fc_net.src.config

import time

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=4096      # MountainCar-v0 uses 200
)

env = gym.make("MountainCarMyEasyVersion-v0")
#env = gym.make("CartPole-v0")
#env = gym.make("Pong-v0")


obs             = env.observation_space
actions_count   = env.action_space.n
env.reset()



model  = models.fc_net.src.model
config = models.fc_net.src.config.Config()

agent = agents.dqn.Agent(env, model, config)


while agent.iterations < 1000000:
    agent.main()

    if agent.iterations%100 == 0:
        env.render()

while True:
    agent.main()
    env.render()
    time.sleep(0.01)
