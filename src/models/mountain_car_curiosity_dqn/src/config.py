import common.decay

class Config(): 

    def __init__(self):
        self.type   = "dqn curiosity"

        self.alpha  = 0.1       #(1.0 - alpha)*q_reward + alpha*curiosity

        self.beta1  = 0.8       #inverse model loss multiplier
        self.beta2  = 0.2       #forward model loss multiplier
        self.beta3  = 0.1       #q-values model loss multiplier

        self.gamma  = 0.95

        self.update_frequency   = 4

        self.batch_size         = 32 
        self.learning_rate      = 0.001
        
        
        self.exploration        = common.decay.Exponential(0.99999, 1.0, 0.1, 0.02)

        self.experience_replay_size = 8192 
 

