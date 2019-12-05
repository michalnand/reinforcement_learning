class Config():

    def __init__(self):
        self.type  = "a2c"
        self.gamma = 0.999
        self.entropy  = 0.00001

        self.update_rate    = 8192
        self.learning_rate  = 0.01
        


