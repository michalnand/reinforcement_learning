import common.decay

class Config(): 

    def __init__(self):
        self.type  = "imitation dueling dqn"
        self.gamma = 0.99

        self.update_frequency = 4

        self.batch_size     = 32 
        self.learning_rate  = 0.0002

        self.experience_replay_size = 16384

        #self.exploration    = common.decay.Linear(1000000, 1.0, 0.05, 0.02)
        #self.expert_decay   = common.decay.Linear(1000000, 1.0, 0.0, 0.0)
        
        #few steps with random exploration
        self.exploration    = common.decay.Linear(500000, 1.0, 0.05, 0.02)

        '''
        2*10^6 steps for training from expert Bilbo
        use 1.0 -> full trust to expert
        after 2*10^6 set 0.0 -> full trust to trauned DQN
        '''
        self.expert_decay   = common.decay.Step(2000000, 1.0, 0.0, 0.0)
        
 
 

