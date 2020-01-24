import agents.dqn

import numpy
import time

import models.cart.src.model
import models.cart.src.config

import common.env_cart

model  = models.cart.src.model
config = models.cart.src.config.Config()

save_path = "./models/cart/"


env = common.env_cart.EnvCart(True)
env.reset()


agent = agents.dqn.Agent(env, model, config, save_path)

score_best = -10000.0
while agent.iterations < 1000000:
    agent.main()    
    if agent.iterations%100000 == 0:
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
while agent.iterations  < 100000:
    agent.main()

print("testing done")
