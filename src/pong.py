import common.env_atari
import agents.dqn

import common.env_pong

env = common.env_pong.Create(size= 64)



import models.generic_net.src.model
import models.generic_net.src.config

model  = models.generic_net.src.model
config = models.generic_net.src.config.Config()

agent = agents.dqn.Agent(env, model, config)

save_path = "./models/generic_net/"
 
while agent.iterations < 1000000:
    agent.main()

    if agent.iterations%1024 == 0:
        agent._print() 
        #env.render()

    if agent.iterations%100000 == 0:
        agent.save(save_path)

agent.save(save_path)


print("program done")