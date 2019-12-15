import gym
import argparse


env = gym.make('gym_go:go-v0', size=9)
done = False
while not done:
    action = env.uniform_random_action()
    observation, reward, done, _ = env.step(action)

    env.render()

    if reward != 0:
        print("reward = ", reward)