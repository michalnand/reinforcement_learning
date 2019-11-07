import cv2
import gym
from gym import spaces

import numpy as np

from matplotlib import pyplot as plt

cv2.ocl.setUseOpenCL(False)

class SetDimensions(gym.Wrapper):
    def __init__(self, env=None, width = 96, height = 96, frame_stacking = 4):
        super(SetDimensions, self).__init__(env)
        self.width  = width
        self.height = height
        self.frame_stacking = frame_stacking

        self.actions_count   = env.action_space.n
        self.shape           = (1, self.frame_stacking, self.height, self.width)

class SkipFrames(gym.Wrapper):
    def __init__(self, env=None, skip = 2):
        super(SkipFrames, self).__init__(env)
        self.skip = skip

    def step(self, action):
        reward_sum = 0.0
        for _ in range(self.skip):
            observation, reward, done, info = self.env.step(action)
            reward_sum+= reward

        return observation, reward_sum, done, info

class FireReset(gym.Wrapper):
    def __init__(self, env=None):
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

        
 

class ResizeFrameEnv(gym.ObservationWrapper):
    def __init__(self, env, width = 96, height = 96):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        
    def observation(self, frame):
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        result = cv2.resize(result, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return result


class FrameStack(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.observation_space = spaces.Box(low=0, high=1.0, shape=(1, self.frame_stacking, self.height, self.width), dtype=np.float32)

    
    def reset(self):
        observation = self.env.reset()
        self.slices = np.zeros(self.shape)
        for i in range(0, self.frame_stacking):
            self.slices[0][i] = observation


        return self.slices

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        for i in reversed(range(self.frame_stacking-1)):
            self.slices[0][i+1] = self.slices[0][i].copy()
        
        self.slices[0][0] = np.array(observation).copy()/255.0
            
        return self.slices, reward, done, info


class Reward(gym.Wrapper):
    def __init__(self, env=None):
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
    frames = np.zeros((observation.shape[1], observation.shape[3],  observation.shape[2]))

    for frame in range(observation.shape[1]):
        for y in range(observation.shape[3]):
            for x in range(observation.shape[2]):
                frames[frame][y][x] = observation[0][frame][y][x]

    f, axarr = plt.subplots(2,2)
    
    axarr[0,0].imshow(frames[0], cmap='gray')
    axarr[0,1].imshow(frames[1], cmap='gray')
    axarr[1,0].imshow(frames[2], cmap='gray')
    axarr[1,1].imshow(frames[3], cmap='gray')

    #plt.imshow(frames[0], interpolation='none')
    
    plt.show() 


def Create(env, width = 96, height = 96, frame_stacking = 4):
    env = SetDimensions(env, width, height, frame_stacking)
    env = SkipFrames(env)
    env = FireReset(env)

    env = ResizeFrameEnv(env)
    env = FrameStack(env)

    env = Reward(env)

    return env
    


