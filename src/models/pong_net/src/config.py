
class Config():

    def __init__(self):
        self.type  = "dqn"
        self.gamma = 0.99

        self.batch_size = 64
        self.learning_rate = 0.001

        self.epsilon        = 1.0
        self.epsilon_end    = 0.1
        self.epsilon_decay  = 0.99999
        self.experience_replay_size = 8192
