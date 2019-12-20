import time
import gym
import common.atari_wrapper
import agents.dqn

'''
#Breakout
import models.atari_dqn.breakout.src.model
import models.atari_dqn.breakout.src.config

model  = models.atari_dqn.breakout.src.model
config = models.atari_dqn.breakout.src.config.Config()
save_path = "./models/atari_dqn/breakout/"
env = gym.make("BreakoutNoFrameskip-v4") 
'''


'''
#Enduro
import models.atari_dqn.enduro.src.model
import models.atari_dqn.enduro.src.config

model  = models.atari_dqn.enduro.src.model
config = models.atari_dqn.enduro.src.config.Config()
save_path = "./models/atari_dqn/enduro/"
env = gym.make("EnduroNoFrameskip-v4") 
'''

'''
#Invaders
import models.atari_dqn.invaders.src.model
import models.atari_dqn.invaders.src.config

model  = models.atari_dqn.invaders.src.model
config = models.atari_dqn.invaders.src.config.Config()
save_path = "./models/atari_dqn/invaders/"
env = gym.make("SpaceInvadersNoFrameskip-v4") 
'''

'''
#KungFuMaster
import models.atari_dqn.kungfumaster.src.model
import models.atari_dqn.kungfumaster.src.config

model  = models.atari_dqn.kungfumaster.src.model
config = models.atari_dqn.kungfumaster.src.config.Config()
save_path = "./models/atari_dqn/kungfumaster/"
env = gym.make("KungFuMasterNoFrameskip-v4") 
'''


#Pacman
import models.atari_dqn.pacman.src.model
import models.atari_dqn.pacman.src.config

model  = models.atari_dqn.pacman.src.model
config = models.atari_dqn.pacman.src.config.Config()
save_path = "./models/atari_dqn/pacman/"
env = gym.make("MsPacmanNoFrameskip-v4") 


'''
#Pong
import models.atari_dqn.pong.src.model
import models.atari_dqn.pong.src.config

model  = models.atari_dqn.pong.src.model
config = models.atari_dqn.pong.src.config.Config()
save_path = "./models/atari_dqn/pong/"
env = gym.make("PongNoFrameskip-v4") 
'''

'''
#Qbert
import models.atari_dqn.qbert.src.model
import models.atari_dqn.qbert.src.config

model  = models.atari_dqn.qbert.src.model
config = models.atari_dqn.qbert.src.config.Config()
save_path = "./models/atari_dqn/qbert/"
env = gym.make("QbertNoFrameskip-v4") 
'''

'''
#seaquest
import models.atari_dqn.seaquest.src.model
import models.atari_dqn.seaquest.src.config

model  = models.atari_dqn.seaquest.src.model
config = models.atari_dqn.seaquest.src.config.Config()
save_path = "./models/atari_dqn/seaquest/"
env = gym.make("SeaquestNoFrameskip-v4") 
'''

env = common.atari_wrapper.Create(env, 96, 96, 4) 


agent = agents.dqn.Agent(env, model, config, save_path, save_stats=False)
agent.load()
agent.disable_training()

while True:
    agent.main()
    env.render()
    time.sleep(1.0/50.0)