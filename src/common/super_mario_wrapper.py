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


from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT



class SetDimensions(gym.Wrapper):
    def __init__(self, env=None, width = 96, height = 96, frame_stacking = 4):
        super(SetDimensions, self).__init__(env)
        self.width  = width
        self.height = height
        self.frame_stacking = frame_stacking

        self.actions_count   = env.action_space.n
        self.shape           = (self.frame_stacking, self.height, self.width)




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
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


class RewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        if done:
            if info["flag_get"]:
                reward+= 50.0
            else:
                reward+= -50.0
 
        reward = reward/10.0 
        return obs, reward, done, info

'''
class LiveLostReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.round_done = True
        self.game_done = True
        self.lives_current = 3

    
    def reset(self):
        if self.game_done:
            observation = self.env.reset()

            self.lives_current = 3
            self.round_done = False
            self.game_done = False
        else:
            observation, _, _, _ = self.env.step(0)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        self.game_done = done

        lives = info["life"]
        if lives < self.lives_current:
            self.lives_current = lives
            reward = -1.0
            self.round_done = True
        else:
            self.round_done = False

        return observation, reward, [self.round_done, self.game_done], info
'''


class LiveLostReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
     

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, [done, done], info


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
            self.slices[i] = ob

        return self.get_state()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        for i in reversed(range(self.frame_stacking-1)):
            self.slices[i+1] = self.slices[i].copy()
        
        self.slices[0] = np.array(ob).copy()
            
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




def Create(env, width = 96, height = 96, frame_stacking = 4, dummy_moves = 1024):
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SetDimensions(env, width, height, frame_stacking)
    env = NoopResetEnv(env)
    env = SkipEnv(env, 4)
    env = RewardEnv(env)
    env = LiveLostReward(env)
    env = ResizeFrameEnv(env)
    env = FrameStack(env)
    env = MakeTensorEnv(env)

    env.observation_space.shape = (env.shape[0], env.shape[1], env.shape[2])

    env.reset()

    actions_count     = env.action_space.n
    for i in range(dummy_moves):
        action = np.random.randint(actions_count)
        _, _, done, _ = env.step(action)

        if done[0]:
            env.reset() 

    env.reset()
    env.reset()
    env.reset()
 

    return env
    

