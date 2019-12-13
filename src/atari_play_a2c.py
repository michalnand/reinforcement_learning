import time
import gym
import common.atari_wrapper
import agents.a2c


#Pong
import models.atari_pong_a2c.src.model
import models.atari_pong_a2c.src.config

model  = models.atari_pong_a2c.src.model
config = models.atari_pong_a2c.src.config.Config()
save_path = "./models/atari_pong_a2c/"
env = gym.make("PongNoFrameskip-v4") 



env = common.atari_wrapper.Create(env, 96, 96, 4) 


agent = agents.a2c.Agent(env, model, config, save_path, save_stats=False)
agent.load()
agent.disable_training()

while True:
    agent.main()
    env.render()
    time.sleep(1.0/50.0)