import gym
import numpy
import time

gym.envs.register(
    id='MountainCarCustom-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=4096      # MountainCar-v0 uses 200
)

env = gym.make("MountainCarCustom-v0")


class SetRewardRange(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward < 0:
            reward = -0.01

        if done: 
            reward = 1.0
        
        return obs, reward, done, info


env = SetRewardRange(env)
env.reset()

obs             = env.observation_space
actions_count   = env.action_space.n


print(obs, actions_count)


while True:
    action = numpy.random.randint(actions_count)
    observation, reward, done, info = env.step(action)
    env.render()

    if reward != 0:
        print("reward = ", reward)
    if done:
        env.reset()

    time.sleep(0.1)
