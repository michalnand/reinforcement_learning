class Config():

    def __init__(self):
        self.type           = "a2c"
        
        self.gamma          = 0.99
        self.learning_rate  = 0.001

        self.entropy_beta   = 0.01
        self.im_beta        = 1.0

        self.batch_size     = 64
        


