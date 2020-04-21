import torch
import torch.nn as nn

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        

        self.actor_features_layers = [
                                        nn.Linear(input_shape[0], 128),
                                        nn.ReLU(),
                                    ]

        self.actor_mu_layers = [        
                                    nn.Linear(128, 64),
                                    nn.ReLU(),                            
                                    nn.Linear(64, outputs_count),
                                    nn.Tanh()     
                                ] 

        self.actor_var_layers = [ 
                                    nn.Linear(128, 64),
                                    nn.ReLU(),  
                                    nn.Linear(64, outputs_count),
                                    nn.Softplus()     
                                ]

        
        self.critic_layers = [ 
                                nn.Linear(input_shape[0], 64),
                                nn.ReLU(),  
                                nn.Linear(64, 64),
                                nn.ReLU(),      
                                
                                nn.Linear(64, 1)
                            ] 


        self.model_features = nn.Sequential(*self.actor_features_layers) 
        self.model_features.to(self.device)

        self.model_mu = nn.Sequential(*self.actor_mu_layers) 
        self.model_mu.to(self.device)

        self.model_var = nn.Sequential(*self.actor_var_layers)
        self.model_var.to(self.device)

        self.model_critic = nn.Sequential(*self.critic_layers)
        self.model_critic.to(self.device)

        print(self.model_features)
        print(self.model_mu)
        print(self.model_var)
        print(self.model_critic)


    def forward(self, state):
        features        = self.model_features(state)
        mu_output       = self.model_mu(features)
        var_output      = self.model_var(features)

        critic_output   = self.model_critic(state)

        return mu_output, var_output, critic_output
    
    def save(self, path):
        print("saving to ", path)

        torch.save(self.model_features.state_dict(), path + "trained/model_features.pt")
        torch.save(self.model_mu.state_dict(), path + "trained/model_mu.pt")
        torch.save(self.model_var.state_dict(), path + "trained/model_var.pt")
        torch.save(self.model_critic.state_dict(), path + "trained/model_critic.pt")
  
    def load(self, path):       
        print("loading from ", path)

        self.model_features.load_state_dict(torch.load(path + "trained/model_features.pt", map_location = self.device))
        self.model_mu.load_state_dict(torch.load(path + "trained/model_mu.pt", map_location = self.device))
        self.model_var.load_state_dict(torch.load(path + "trained/model_var.pt", map_location = self.device))
        self.model_critic.load_state_dict(torch.load(path + "trained/model_critic.pt", map_location = self.device))
    
        self.model_features.eval() 
        self.model_mu.eval() 
        self.model_var.eval() 
        self.model_critic.eval()  
    
