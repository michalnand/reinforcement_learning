import time
import gym
import gym_super_mario_bros
import common.super_mario_wrapper

import agents.dqn
import agents.a2c


import models.super_mario.dqn.src.model
import models.super_mario.dqn.src.config
import models.super_mario.a2c.src.model
import models.super_mario.a2c.src.config


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = common.super_mario_wrapper.Create(env, dummy_moves = 0)
env.reset()

'''
model       = models.super_mario.dqn.src.model
config      = models.super_mario.dqn.src.config.Config()
save_path   = "./models/super_mario/dqn/"
agent = agents.dqn.Agent(env, model, config, save_path, save_stats=False)
'''

model       = models.super_mario.a2c.src.model
config      = models.super_mario.a2c.src.config.Config()
save_path   = "./models/super_mario/a2c/"
agent = agents.a2c.Agent(env, model, config, save_path, save_stats=False)


agent.load()
agent.disable_training()

while True:
    agent.main()
    env.render()
    time.sleep(1.0/50.0)