class Config(): 
    def __init__(self):
        self.gamma_ext              = 0.0
        self.gamma_int              = 0.99

        self.ext_adv_coeff          = 0.0
        self.int_adv_coeff          = 1.0

        self.entropy_beta           = 0.001
        self.eps_clip               = 0.1

        self.steps                  = 128
        self.batch_size             = 4
        
        self.training_epochs        = 4
        self.envs_count             = 32
        
        self.learning_rate_ppo      = 0.0002
        self.learning_rate_rnd      = 0.0001

        self.normalise_state_std    = False
        self.normalise_im_std       = False
        
