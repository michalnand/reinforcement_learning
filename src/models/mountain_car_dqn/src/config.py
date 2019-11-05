import common.decay

class Config():

    def __init__(self):
        self.type  = "dqn"
        self.gamma = 0.999

        self.batch_size     = 32
        self.learning_rate  = 0.001

        self.epsilon        = common.decay.Exponential(0.99999, 1.0, 0.1, 0.02)
        
        self.experience_replay_size = 256 # 8192


