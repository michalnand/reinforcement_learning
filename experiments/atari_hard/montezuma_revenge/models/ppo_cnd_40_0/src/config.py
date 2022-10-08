class Config(): 
    def __init__(self):
        self.gamma_ext              = 0.998
        self.gamma_int              = 0.99

        self.ext_adv_coeff          = 2.0
        self.int_adv_coeff          = 1.0

        self.int_reward_coeff       = 0.25
        self.regularisation_coeff   = 0.0
        self.symmetry_loss_coeff    = 1.0 
        self.cnd_dropout            = 0.75

        self.entropy_beta           = 0.001
        self.eps_clip               = 0.1

        self.steps                  = 128
        self.batch_size             = 4
        
        self.training_epochs        = 4
        self.envs_count             = 128

        self.learning_rate_ppo          = 0.0001 
        self.learning_rate_cnd          = 0.0001
        self.learning_rate_cnd_target   = 0.0001 

        self.cnd_regularisation_loss    = "mse"
        self.ppo_symmetry_loss          = "actions"
    
        self.normalise_state_mean       = True
        self.normalise_state_std        = True
        