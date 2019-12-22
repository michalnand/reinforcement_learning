import common.decay

class Config(): 

    def __init__(self):
        self.type  = "imitation dueling dqn"
        self.gamma = 0.99

        self.update_frequency = 4

        self.batch_size     = 32 
        self.learning_rate  = 0.0002

        self.experience_replay_size = 16384

        self.exploration    = common.decay.LinearDelayed(self.experience_replay_size*10, 1000000, 0.5, 0.05, 0.02)
        self.expert_decay   = common.decay.Linear(self.experience_replay_size*10, 1.0, 0.0, 0.0)
        

 
 

