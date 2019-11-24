import common.decay

class Config(): 

    def __init__(self):
        self.type  = "dqn"
        self.gamma = 0.99

        self.batch_size     = 32 
        #self.learning_rate  = 0.0002
        self.learning_rate  = 0.001

        #self.exploration    = common.decay.Linear(2000000, 1.0, 0.02, 0.02)
        self.exploration   = common.decay.Exponential(0.999999, 1.0, 0.01, 0.01)

        self.experience_replay_size = 10000 
 

