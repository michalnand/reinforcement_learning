import gym_super_mario_bros
import common.super_mario_wrapper

import numpy
import time

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = common.super_mario_wrapper.Create(env, dummy_moves = 256)

env.reset()




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

actions_count     = env.action_space.n

while True:
    action = numpy.random.randint(actions_count)
    state, reward, done, info = env.step(action)
    env.render()
    
    #observation_show(state)

    
    if reward != 0:
        print("reward = ", reward)

    if done[1]:
        print("\n\nGAME DONE\n\n")
        env.reset()
        time.sleep(1.0)


    time.sleep(0.02)

