import time
import gym
import common.atari_wrapper
import agents.a2c

'''
#Breakout
import models.atari_a2c.breakout.src.model
import models.atari_a2c.breakout.src.config

model  = models.atari_a2c.breakout.src.model
config = models.atari_a2c.breakout.src.config.Config()
save_path = "./models/atari_a2c/breakout/"
env = gym.make("BreakoutNoFrameskip-v4") 
'''

'''
#Invaders
import models.atari_a2c.invaders.src.model
import models.atari_a2c.invaders.src.config

model  = models.atari_a2c.invaders.src.model
config = models.atari_a2c.invaders.src.config.Config()
save_path = "./models/atari_a2c/invaders/"
env = gym.make("SpaceInvadersNoFrameskip-v4") 
'''

'''
#KungFuMaster
import models.atari_a2c.kungfumaster.src.model
import models.atari_a2c.kungfumaster.src.config

model  = models.atari_a2c.kungfumaster.src.model
config = models.atari_a2c.kungfumaster.src.config.Config()
save_path = "./models/atari_a2c/kungfumaster/"
env = gym.make("KungFuMasterNoFrameskip-v4") 
'''

'''
#Pacman
import models.atari_a2c.pacman.src.model
import models.atari_a2c.pacman.src.config

model  = models.atari_a2c.pacman.src.model
config = models.atari_a2c.pacman.src.config.Config()
save_path = "./models/atari_a2c/pacman/"
env = gym.make("MsPacmanNoFrameskip-v4") 
'''

'''
#Pong
import models.atari_a2c.pong.src.model
import models.atari_a2c.pong.src.config

model  = models.atari_a2c.pong.src.model
config = models.atari_a2c.pong.src.config.Config()
save_path = "./models/atari_a2c/pong/"
env = gym.make("PongNoFrameskip-v4") 
'''

'''
#QBert
import models.atari_a2c.qbert.src.model
import models.atari_a2c.qbert.src.config

model  = models.atari_a2c.qbert.src.model
config = models.atari_a2c.qbert.src.config.Config()
save_path = "./models/atari_a2c/qbert/"
env = gym.make("QbertNoFrameskip-v4") 
'''

#Seaquest
import models.atari_a2c.seaquest.src.model
import models.atari_a2c.seaquest.src.config

model  = models.atari_a2c.seaquest.src.model
config = models.atari_a2c.seaquest.src.config.Config()
save_path = "./models/atari_a2c/seaquest/"
env = gym.make("SeaquestNoFrameskip-v4") 


env = common.atari_wrapper.Create(env, 96, 96, 4) 

agent = agents.a2c.Agent(env, model, config, save_path, save_stats=False)
agent.load()
agent.disable_training()

while True:
    agent.main()
    env.render()
    time.sleep(1.0/50.0)