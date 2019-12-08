class Config():

    def __init__(self):
        self.type           = "a2c"
        self.gamma          = 0.999
        self.entropy_beta   = 0.01

        self.batch_size     = 256
        self.learning_rate  = 0.01
        


