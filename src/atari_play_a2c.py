import time
import gym
import common.atari_wrapper
import agents.a2c

import models.atari_a2c.pong_p.src.model
import models.atari_a2c.pong_p.src.config


save_path = "./models/atari_a2c/pong_p/"
paralel_envs_count = 1

envs = [] 

for _ in range(paralel_envs_count):
    env = gym.make("PongNoFrameskip-v4")
    env = common.atari_wrapper.Create(env)
    env.reset()

    envs.append(env)

obs             = envs[0].observation_space
actions_count   = envs[0].action_space.n

model  = models.atari_a2c.pong_p.src.model
config = models.atari_a2c.pong_p.src.config.Config()
 
agent = agents.a2c.Agent(envs, model, config, save_path, save_stats=False)


agent.load()
agent.disable_training()

while True:
    agent.main()
    env.render()
    time.sleep(1.0/50.0)