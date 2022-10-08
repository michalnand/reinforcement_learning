import gym
import numpy
import time

import RLAgents

import models.dqn_baseline.src.model            as Model
import models.dqn_baseline.src.config           as Config


path = "models/dqn_baseline/"

env = gym.make("BreakoutNoFrameskip-v4")

env = RLAgents.WrapperAtari(env)
env.reset()

config = Config.Config()

agent = RLAgents.AgentDQN(env, Model, config)

max_iterations = 16*(10**6) 

#trainig = RLAgents.TrainingIterations(env, agent, max_iterations, path, 10000)
#trainig.run() 


agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main(False)

    env.render()
    time.sleep(0.01)
