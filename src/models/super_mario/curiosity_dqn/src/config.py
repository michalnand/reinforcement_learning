import common.decay

class Config(): 

    def __init__(self):
        self.type   = "dqn curiosity"

        self.alpha  = 1.0       #(1.0 - alpha)*q_reward + alpha*curiosity
        self.beta   = 0.2       #(1.0 - beta)*loss_inverse + beta*loss_forward
        self.gamma  = 0.95


        self.update_frequency   = 4

        self.batch_size         = 32 
        self.dqn_learning_rate  = 0.0001
        self.icm_learning_rate  = 0.001
        
        #self.exploration    = common.decay.Linear(1000000, 1.0, 0.1, 0.02)
        self.exploration     = common.decay.Exponential(0.999999, 1.0, 0.1, 0.02)

        self.experience_replay_size = 16384
 

