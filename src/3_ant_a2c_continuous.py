import gym
import pybulletgym
import numpy
import agents.a2c_continuous
import models.ant_a2c_continuous.src.model
import models.ant_a2c_continuous.src.config

save_path = "./models/ant_a2c_continuous/"

paralel_envs_count = 8

envs = [] 

for i in range(paralel_envs_count):
    env = gym.make("AntPyBulletEnv-v0")
    obs = env.reset()
    env.seed(i)

    envs.append(env)




obs             = envs[0].observation_space
actions_size    = envs[0].action_space.shape[0]


model  = models.ant_a2c_continuous.src.model
config = models.ant_a2c_continuous.src.config.Config()
 
agent = agents.a2c_continuous.Agent(envs, model, config, save_path)


score_best = -10000.0
while agent.iterations < 2000000:
    agent.main()    
    if agent.iterations%10000 == 0:
        if agent.training_stats.game_score_smooth > score_best:
            score_best = agent.training_stats.game_score_smooth
            agent.save()
            
            print("\n\n\n")
            print("saving best agent")
            print("iteration = ", agent.iterations)
            print("score_best = ", score_best)
            print("\n\n\n")
            

print("training done")

agent.load()
agent.disable_training()

agent.iterations = 0
while agent.iterations  < 1000000:
    agent.main()

print("testing done")




'''
 

paralel_envs_count = 1

envs = [] 

for _ in range(paralel_envs_count):
    env = gym.make("AntPyBulletEnv-v0")
    env.render()
    env.reset()

    envs.append(env)

obs             = envs[0].observation_space
actions_size    = envs[0].action_space.shape[0]

model  = models.ant_a2c_continuous.src.model
config = models.ant_a2c_continuous.src.config.Config()
 
agent = agents.a2c_continuous.Agent(envs, model, config, save_path, save_stats=False)


agent.load()
agent.disable_training()

while True:
    agent.main()

'''