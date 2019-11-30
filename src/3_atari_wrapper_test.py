import gym
import numpy
import time

import common.atari_wrapper

from matplotlib import pyplot as plt


def observation_show(observation):

    shape = observation.shape

    print(shape)
    frames = numpy.zeros((shape[0], shape[1],  shape[2]))

    print("observation_show ", frames.shape)

    for frame in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                frames[frame][y][x] = observation[frame][y][x]

    f, axarr = plt.subplots(2,2)
    
    axarr[0,0].imshow(frames[0], cmap='gray')
    axarr[0,1].imshow(frames[1], cmap='gray')
    axarr[1,0].imshow(frames[2], cmap='gray')
    axarr[1,1].imshow(frames[3], cmap='gray')

    #plt.imshow(frames[0], interpolation='none')
    
    plt.show() 


#env = gym.make("BreakoutNoFrameskip-v4")
#env = gym.make("EnduroNoFrameskip-v4")
#env = gym.make("SpaceInvadersNoFrameskip-v4")
env = gym.make("KungFuMasterNoFrameskip-v4")
#env = gym.make("MsPacmanNoFrameskip-v4")
#env = gym.make("PongNoFrameskip-v4")
#env = gym.make("QbertNoFrameskip-v4") 
#env = gym.make("SeaquestNoFrameskip-v4") 


env = common.atari_wrapper.Create(env)

env.reset()



obs             = env.observation_space
actions_count   = env.action_space.n


print("ENV info ", obs.shape, actions_count)

frames = 0
while True:
    action = numpy.random.randint(actions_count)
    observation, reward, done, info = env.step(action)
    env.render()

    '''
    if frames > 100:
        observation_show(observation)    
    '''
    frames+= 1

    if reward != 0:
        print("reward = ", reward)

    if done[0]:
        print("round done\n\n")
        env.reset()
        time.sleep(1.0)


    if done[1]:
        print("\n\nGAME DONE\n\n")
        env.reset()
        time.sleep(1.0)


    time.sleep(0.01)
