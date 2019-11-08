import cv2
import gym
from gym import spaces

import numpy as np

from matplotlib import pyplot as plt

cv2.ocl.setUseOpenCL(False)

class SetDimensions(gym.Wrapper):
    def __init__(self, env, width = 96, height = 96, frame_stacking = 4):
        #gym.Wrapper.__init__(self, env)
        super(SetDimensions, self).__init__(env)
        self.width  = width
        self.height = height
        self.frame_stacking = frame_stacking

        self.observation_space = spaces.Box(low=0, high=1.0, shape=(1, self.frame_stacking, self.height, self.width), dtype=np.float32)

        print("SetDimensions")



class SkipFrames(gym.Wrapper):
    def __init__(self, env, skip = 2):
        #gym.Wrapper.__init__(self, env)
        super(SkipFrames, self).__init__(env)
        self.skip = skip

    def step(self, action):
        reward_sum = 0.0
        for _ in range(self.skip):
            observation, reward, done, info = self.env.step(action)
            reward_sum+= reward

        return observation, reward_sum, done, info

class FireReset(gym.Wrapper):
    def __init__(self, env):
        #gym.Wrapper.__init__(self, env)
        super(FireReset, self).__init__(env)
        

    def reset(self):
        self.env.reset()
        observation, _, done, _ = self.env.step(1)
        
        if done:
            self.env.reset()

        observation, _, done, _ = self.env.step(2)

        if done:
            self.env.reset()

        return observation

        
 

class ResizeFrame(gym.Wrapper):
    def __init__(self, env):
        #gym.Wrapper.__init__(self, env)
        super(ResizeFrame, self).__init__(env)

    def reset(self):
        return self.resize(self.env.reset())

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        return self.resize(observation), reward, done, info

    def resize(self, frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        
        #to grayscale
        grayscale = (img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114)/255.0

        resized_screen = cv2.resize(grayscale, (self.width, self.height), interpolation=cv2.INTER_AREA)
        result = np.reshape(resized_screen, (1, self.height, self.width) )
        return result


class FrameStack(gym.Wrapper):
    def __init__(self, env):
        #gym.Wrapper.__init__(self, env)
        super(FrameStack, self).__init__(env)

    
    def reset(self):
        observation = self.env.reset()
        self.slices = np.zeros((1, self.frame_stacking, self.height, self.width))
        for i in range(0, self.frame_stacking):
            self.slices[0][i] = observation

        return self.slices

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        for i in reversed(range(self.frame_stacking-1)):
            self.slices[0][i+1] = self.slices[0][i].copy()
        
        self.slices[0][0] = np.array(observation).copy()
            
        return self.slices, reward, done, info


class Reward(gym.Wrapper):
    def __init__(self, env):
        #gym.Wrapper.__init__(self, env)
        super(Reward, self).__init__(env)


    def reset(self):
        observation = self.env.reset()

        self.lives = self.env.unwrapped.ale.lives()

        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if reward > 1.0:
            reward = 1.0

        if reward < -1.0:
            reward = -1.0


        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives:
            self.lives = lives
            reward = -1.0

        return observation, reward, done, info


def observation_show(observation):

    frames = np.zeros((observation.shape[1], observation.shape[2],  observation.shape[3]))

    print("observation_show ", frames.shape)

    for frame in range(observation.shape[1]):
        for y in range(observation.shape[2]):
            for x in range(observation.shape[3]):
                frames[frame][y][x] = observation[0][frame][y][x]

    f, axarr = plt.subplots(2,2)
    
    axarr[0,0].imshow(frames[0], cmap='gray')
    axarr[0,1].imshow(frames[1], cmap='gray')
    axarr[1,0].imshow(frames[2], cmap='gray')
    axarr[1,1].imshow(frames[3], cmap='gray')

    #plt.imshow(frames[0], interpolation='none')
    
    plt.show() 


def Create(env_name, width = 96, height = 96, frame_stacking = 4):
    env = gym.make(env_name)

    env = SetDimensions(env, width, height, frame_stacking)
    env = FireReset(env)
    env = SkipFrames(env)

    env = ResizeFrame(env)
    env = FrameStack(env)

    #env = Reward(env)

    return env
    


