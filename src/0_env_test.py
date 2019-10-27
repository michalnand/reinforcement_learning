import gym
import numpy

env = gym.make("MountainCar-v0")
#env = gym.make("Pong-v0")
#env = gym.make("Breakout-v0")
env.reset()

obs             = env.observation_space
actions_count   = env.action_space.n


print(obs, actions_count)


while True:
    action = numpy.random.randint(actions_count)
    observation, reward, done, info = env.step(action)
    env.render()

    if done:
        env.reset()