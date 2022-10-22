class Config(): 
    def __init__(self):
        self.gamma_ext              = 0.998
        self.gamma_int              = 0.99

        #advanteges weighting
        self.ext_adv_coeff          = 2.0
        self.int_adv_coeff          = 1.0

        #in reward weight, ppo reg weight, cnd dropout rate 0..100%
        self.int_reward_coeff                   = 0.5
        self.ppo_regularization_loss_coeff      = 0.0 
        self.cnd_dropout                        = 0.0

        self.entropy_beta           = 0.001
        self.eps_clip               = 0.1

        self.steps                  = 128
        self.batch_size             = 4
        
        self.training_epochs        = 4
        self.envs_count             = 128

        #learning rates
        self.learning_rate_ppo          = 0.0001 
        self.learning_rate_cnd          = 0.0001
        self.learning_rate_cnd_target   = 0.0001 

        #self supervised learning loss
        self.ppo_regularization_loss    = None
        self.cnd_regularization_loss    = "mse"
        
        #state normalisation for CND internal motivation
        self.normalise_state_mean       = False
        self.normalise_state_std        = False

        #used augmentations
        self.ppo_augmentations          = []
        self.ppo_reg_augmentations      = []
        self.cnd_augmentations          = ["mask", "noise"]
    