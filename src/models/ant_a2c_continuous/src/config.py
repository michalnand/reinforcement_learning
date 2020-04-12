class Config():

    def __init__(self):
        self.type           = "a2c"
        
        self.gamma          = 0.99
        self.learning_rate  = 0.0001

        self.entropy_beta   = 0.1
        self.batch_size     = 128
        


