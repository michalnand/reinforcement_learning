import common.decay

class Config(): 

    def __init__(self):
        self.type   = "dqn curiosity"

        self.alpha  = 0.5       #(1.0 - alpha)*q_reward + alpha*curiosity
        self.beta1  = 0.2       #(1.0 - beta)*loss_inverse + beta*loss_forward
        self.beta2  = 0.1

        self.gamma  = 0.95

        self.update_frequency   = 4

        self.batch_size         = 32 
        self.learning_rate      = 0.001
        
        
        self.exploration    = common.decay.Linear(1000000, 1.0, 0.05, 0.02)

        self.experience_replay_size = 2048  #16384
 

