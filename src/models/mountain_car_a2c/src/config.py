class Config():

    def __init__(self):
        self.type  = "a2c"
        self.gamma = 0.999
        self.entropy_ratio  = 0.001

        self.update_frequency = 32

        self.batch_size     = 32
        self.learning_rate  = 0.0001

        

        
        self.experience_replay_size = 8192


