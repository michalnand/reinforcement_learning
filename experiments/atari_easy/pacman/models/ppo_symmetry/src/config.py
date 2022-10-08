class Config(): 
    def __init__(self):
        self.gamma                  = 0.99
        
        self.entropy_beta           = 0.001
        self.eps_clip               = 0.1

        self.regularisation_coeff   = 0.0001
        self.symmetry_loss_coeff    = 0.001
        self.ppo_symmetry_loss      = "mse"

        self.steps                  = 128
        self.batch_size             = 4
        
        self.training_epochs        = 4
        self.envs_count             = 32
        
        self.learning_rate          = 0.0001
  