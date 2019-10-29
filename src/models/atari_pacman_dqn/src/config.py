
class Config():

    def __init__(self):
        self.type  = "dqn"
        self.gamma = 0.99

        self.batch_size     = 32
        self.learning_rate  = 0.0001
 
        self.epsilon        = 1.0
        self.epsilon_end    = 0.05
        self.epsilon_decay  = 150000
        self.experience_replay_size = 10000
 
