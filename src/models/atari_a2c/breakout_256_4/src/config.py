class Config(): 

    def __init__(self):
        self.type           = "a2c"
        
        self.gamma          = 0.99
        self.entropy_beta   = 0.001
        self.batch_size     = 256 
        self.bellman_steps  = 4

        self.learning_rate  = 0.001
        
 
 
