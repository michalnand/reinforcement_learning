import numpy
from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation


def run_episode(env):
    done = False
    episode_return = 0.0
    
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        episode_return += reward

        print("state shape = ", observation[0].shape)

    return episode_return



eval_seeds = [1001, 1002, 1003, 1004, 1005]
# Create the ObstacleTowerEnv gym and launch ObstacleTower
env = ObstacleTowerEnv("/home/michal/libs/ObstacleTower/obstacletower", retro=False, realtime_mode=True )

# Wrap the environment with the ObstacleTowerEvaluation wrapper
# and provide evaluation seeds.
env = ObstacleTowerEvaluation(env, eval_seeds)

# We can run episodes (in this case with a random policy) until 
# the "evaluation_complete" flag is True.  Attempting to step or reset after
# all of the evaluation seeds have completed will result in an exception.
while not env.evaluation_complete:
    episode_rew = run_episode(env)

# Finally the evaluation results can be fetched as a dictionary from the 
# environment wrapper.
print(env.results)

env.close()



















env.seed(5)
env.floor(15)

config = {'agent-perspective': 10}
state = env.reset(config=config)


print("state shape = ", state.shape)

while True:
    action = env.action_space.sample()

    state, reward, done, info = env.step(action)
    env.render()
    
    
    if done:
        state = env.reset()

    if reward != 0:
        print(reward)


env.close()
