import time
import gym
import common.atari_wrapper
import agents.dqn

'''
#Breakout
import models.atari_breakout_dqn.src.model
import models.atari_breakout_dqn.src.config

model  = models.atari_breakout_dqn.src.model
config = models.atari_breakout_dqn.src.config.Config()
save_path = "./models/atari_breakout_dqn/"
env = gym.make("BreakoutNoFrameskip-v4") 
'''

'''
#Enduro
import models.atari_enduro_dqn.src.model
import models.atari_enduro_dqn.src.config

model  = models.atari_enduro_dqn.src.model
config = models.atari_enduro_dqn.src.config.Config()
save_path = "./models/atari_enduro_dqn/"
env = gym.make("EnduroNoFrameskip-v4") 
'''

'''
#Invaders
import models.atari_invaders_dqn.src.model
import models.atari_invaders_dqn.src.config

model  = models.atari_invaders_dqn.src.model
config = models.atari_invaders_dqn.src.config.Config()
save_path = "./models/atari_invaders_dqn/"
env = gym.make("SpaceInvadersNoFrameskip-v4") 
'''

'''
#KungFuMaster
import models.atari_invaders_dqn.src.model
import models.atari_invaders_dqn.src.config

model  = models.atari_invaders_dqn.src.model
config = models.atari_invaders_dqn.src.config.Config()
save_path = "./models/atari_invaders_dqn/"
env = gym.make("KungFuMasterNoFrameskip-v4") 
'''

'''
#Pacman
import models.atari_pacman_dqn.src.model
import models.atari_pacman_dqn.src.config

model  = models.atari_pacman_dqn.src.model
config = models.atari_pacman_dqn.src.config.Config()
save_path = "./models/atari_pacman_dqn/"
env = gym.make("MsPacmanNoFrameskip-v4") 
'''

'''
#Pong
import models.atari_pong_dqn.src.model
import models.atari_pong_dqn.src.config

model  = models.atari_pong_dqn.src.model
config = models.atari_pong_dqn.src.config.Config()
save_path = "./models/atari_pong_dqn/"
env = gym.make("PongNoFrameskip-v4") 
'''

'''
#Qbert
import models.atari_qbert_dqn.src.model
import models.atari_qbert_dqn.src.config

model  = models.atari_qbert_dqn.src.model
config = models.atari_qbert_dqn.src.config.Config()
save_path = "./models/atari_qbert_dqn/"
env = gym.make("QbertNoFrameskip-v4") 
'''


#seaquest
import models.atari_seaquest_dqn.src.model
import models.atari_seaquest_dqn.src.config

model  = models.atari_seaquest_dqn.src.model
config = models.atari_seaquest_dqn.src.config.Config()
save_path = "./models/atari_seaquest_dqn/"
env = gym.make("SeaquestNoFrameskip-v4") 



env = common.atari_wrapper.Create(env, 96, 96, 4) 


agent = agents.dqn.Agent(env, model, config, save_path, save_stats=False)
agent.load()
agent.disable_training()

while True:
    agent.main()
    env.render()
    time.sleep(1.0/50.0)