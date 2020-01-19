import time
import gym
import common.atari_wrapper


import numpy
from matplotlib import pyplot as plt

env = gym.make("MsPacmanNoFrameskip-v4") 
env = common.atari_wrapper.Create(env, 96, 96, 4) 


#Pacman DQN
import agents.dqn
import models.atari_dqn.pacman.src.model
import models.atari_dqn.pacman.src.config

model  = models.atari_dqn.pacman.src.model
config = models.atari_dqn.pacman.src.config.Config()
save_path = "./models/atari_dqn/pacman/"
agent = agents.dqn.Agent(env, model, config, save_path, save_stats=False)
agent.load()
agent.disable_training()


'''
#Pacman Dueling
import agents.dqn
import models.atari_dqn.pacman_dueling_dqn.src.model
import models.atari_dqn.pacman_dueling_dqn.src.config

model  = models.atari_dqn.pacman_dueling_dqn.src.model
config = models.atari_dqn.pacman_dueling_dqn.src.config.Config()
save_path = "./models/atari_dqn/pacman_dueling_dqn/"
agent = agents.dqn.Agent(env, model, config, save_path, save_stats=False)
agent.load()
agent.disable_training()
'''

'''
#Pacman A2C
import agents.a2c
import models.atari_a2c.pacman.src.model
import models.atari_a2c.pacman.src.config

model  = models.atari_a2c.pacman.src.model
config = models.atari_a2c.pacman.src.config.Config()
save_path = "./models/atari_a2c/pacman/"
agent = agents.a2c.Agent(env, model, config, save_path, save_stats=False)
agent.load()
agent.disable_training()
'''

'''
#Pacman DQN Rainbow
import agents.rainbow
import models.atari_dqn.pacman_rainbow.src.model
import models.atari_dqn.pacman_rainbow.src.config

model  = models.atari_dqn.pacman_rainbow.src.model
config = models.atari_dqn.pacman_rainbow.src.config.Config()
save_path = "./models/atari_dqn/pacman_rainbow/"
agent = agents.rainbow.Agent(env, model, config, save_path, save_stats=False)
agent.load()
agent.disable_training()
'''

'''
#Pacman imitation
import agents.dqn_imitation
import models.atari_dqn.pacman_imitation_dueling_dqn.src.model
import models.atari_dqn.pacman_imitation_dueling_dqn.src.config

import  models.atari_dqn.pacman.src.model as ExpertModel


model  = models.atari_dqn.pacman_imitation_dueling_dqn.src.model
config = models.atari_dqn.pacman_imitation_dueling_dqn.src.config.Config()

save_path = "./models/atari_dqn/pacman_imitation_dueling_dqn/"

expert_model = ExpertModel.Model(env.observation_space.shape, env.action_space.n)
expert_model.load("models/atari_dqn/pacman/")

agent = agents.dqn_imitation.Agent(env, model, config, expert_model, save_path, save_stats=False)

agent.load()
agent.disable_training()
'''


def activity_show(observation, activity, alpha = 0.3):

    shape = observation.shape

    image = numpy.zeros((shape[1],  shape[2], 3))


    for y in range(shape[1]):
        for x in range(shape[2]):
            r = observation[0][y][x] + alpha*activity[y][x]
            g = observation[0][y][x]
            b = observation[0][y][x]

            image[y][x][0] = r
            image[y][x][1] = g
            image[y][x][2] = b
    
    
    plt.imshow(image, interpolation='bicubic')
    
    plt.draw()
    plt.pause(0.001)
    #plt.show() 



iteration = 0
while True:
    agent.main()
    env.render()
    time.sleep(1.0/50.0)

    '''
    activity = agent.model.get_activity_map(agent.observation)

    if iteration%10 == 0:
        activity_show(agent.observation, activity, 0.7)
    '''
    iteration+= 1
    