# This file was based on
# https://github.com/openai/baselines/blob/edb52c22a5e14324304a491edc0f91b6cc07453b/baselines/common/atari_wrappers.py
# its license:
#
# The MIT License
#
# Copyright (c) 2017 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces

from matplotlib import pyplot as plt

#cv2.ocl.setUseOpenCL(False)

class SetDimensions(gym.Wrapper):
    def __init__(self, env=None, width = 96, height = 96, frame_stacking = 4):
        super(SetDimensions, self).__init__(env)
        self.width  = width
        self.height = height
        self.frame_stacking = frame_stacking

        self.actions_count   = env.action_space.n
        self.shape           = (1, self.frame_stacking, self.height, self.width)



class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)

        return obs



class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)

        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = np.clip(reward, -10.0, 10.0)
        return obs, reward, done, info

class ResizeFrameEnv(gym.ObservationWrapper):
    def __init__(self, env, width = 96, height = 96):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        
    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


class FrameStack(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def reset(self):
        ob = self.env.reset()
        self.slices = np.zeros(self.shape)
        for i in range(0, self.frame_stacking):
            self.slices[0][i] = ob

        return self.get_state()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        for i in reversed(range(self.frame_stacking-1)):
            self.slices[0][i+1] = self.slices[0][i].copy()
        
        self.slices[0][0] = np.array(ob).copy()
            
        return self.get_state(), reward, done, info

    def get_state(self):
        return self.slices


class MakeTensorEnv(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        #swaped = np.moveaxis(observation, 0, 0)
        #result = np.reshape(swaped, (1, observation.shape[2], self.height, self.width))
        result = observation/255.0
        return result



def observation_show(observation):
    image = np.zeros((observation.shape[3],  observation.shape[2], 3))

    for ch in range(3):
        for y in range(observation.shape[3]):
            for x in range(observation.shape[2]):
                image[y][x][ch] = observation[0][0][y][x]

    plt.imshow(image, interpolation='none')
    plt.show()


def Create(env, width = 96, height = 96, frame_stacking = 4):
    env = SetDimensions(env, width, height, frame_stacking)
    env = NoopResetEnv(env)
    env = FireResetEnv(env)
    env = MaxAndSkipEnv(env)
    env = ClipRewardEnv(env)
    env = ResizeFrameEnv(env)
    env = FrameStack(env)
    env = MakeTensorEnv(env)

    env.observation_space.shape = (env.shape[1], env.shape[2], env.shape[3])

    return env
    


