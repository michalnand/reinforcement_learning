import gym
import common.atari_wrapper
import common.env_multi


import agents.dqn


import numpy
import time

from matplotlib import pyplot as plt

import models.multi_agent.atari_dqn_2.src.model
import models.multi_agent.atari_dqn_2.src.config



model  = models.multi_agent.atari_dqn_2.src.model
config = models.multi_agent.atari_dqn_2.src.config.Config()

save_path = "./models/multi_agent/atari_dqn_2/" 



width  = 96
height = 96
frame_stacking = 4

envs = []

envs.append(common.atari_wrapper.Create(gym.make("BreakoutNoFrameskip-v4"), width, height, frame_stacking))
envs.append(common.atari_wrapper.Create(gym.make("MsPacmanNoFrameskip-v4"), width, height, frame_stacking))
envs.append(common.atari_wrapper.Create(gym.make("SeaquestNoFrameskip-v4"), width, height, frame_stacking))
envs.append(common.atari_wrapper.Create(gym.make("QbertNoFrameskip-v4"), width, height, frame_stacking))


env = common.env_multi.EnvMulti(envs, 512)
env.reset()


active_env = 1

env.set_env(active_env)



agent = agents.dqn.Agent(env, model, config, save_path, save_stats=False)
agent.load()
agent.disable_training()





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
    plt.pause(0.01)
    #plt.show() 
    
iteration = 0


while True:
    agent.main()
    env.render()
    time.sleep(1.0/50.0)

    iteration+= 1

    if iteration%512 == 0:
        active_env = (active_env + 1)%env.get_env_count()

        env.set_env(active_env)

    '''
    if iteration%20 == 0:
        activity = agent.model.get_activity_map(agent.observation)
        activity_show(agent.observation, activity, 0.7)
    '''