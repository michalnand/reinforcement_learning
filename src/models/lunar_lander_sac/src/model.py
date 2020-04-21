import torch
import torch.nn as nn


class SoftQNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_count = 256, init_weight_range = 0.003):
        super(SoftQNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [ 
                        nn.Linear(input_dim + output_dim, hidden_count),
                        nn.ReLU(),                     
                        nn.Linear(hidden_count, hidden_count),
                        nn.ReLU(),    
                        nn.Linear(hidden_count, 1)                 
                    ]

        #TODO : check xavier initialisation scheme
        '''
        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        torch.nn.init.xavier_uniform_(self.layers[4].weight)
        '''

        torch.nn.init.uniform_(self.layers[0].weight, -init_weight_range, init_weight_range)
        torch.nn.init.uniform_(self.layers[2].weight, -init_weight_range, init_weight_range)
        torch.nn.init.uniform_(self.layers[4].weight, -init_weight_range, init_weight_range)
 
        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

    def forward(self, state, action)
        x = torch.cat([state, action], dim = 1)
        return self.model(x) 

    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/soft_q.pt")
      
    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "trained/soft_q.pt", map_location = self.device))



class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_count = 256, init_weight_range = 0.003, log_std_min = -20.0, log_std_max=2:
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.features_layers = [ 
                                    nn.Linear(input_dim, hidden_count),
                                    nn.ReLU(),                   
                                    nn.Linear(hidden_count, hidden_count),
                                    nn.ReLU()                   
                                ]

        self.mu_layers =    [
                                nn.Linear(hidden_count, output_dim)
                            ]

        self.log_std_layers =   [
                                    nn.Linear(hidden_count, output_dim)
                                ]


        self.features_model = nn.Sequential(*self.features_layers)
        self.features_model.to(self.device)
        
        torch.nn.init.uniform_(self.mu_layers[0].weight, -init_weight_range, init_weight_range)
        self.mu_model = nn.Sequential(*self.mu_layers)
        self.mu_model.to(self.device)


        torch.nn.init.uniform_(self.log_std_layers[0].weight, -init_weight_range, init_weight_range)
        self.log_std_model = nn.Sequential(*self.log_std_layers)
        self.log_std_model.to(self.device)


    def forward(self, state):
        features = self.features_model(state)
        
        mu      = self.mu_model(features)

        log_std = self.log_std_model(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, eps=10.0**-6):
        mu, log_std = self.forward(state)
        
        std         = log_std.exp()

        normal = torch.distributions.Normal(mu, std)
        z      = normal.rsample()
        action = torch.tanh(z)

        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + eps)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi

    def save(self, path):
        torch.save(self.features_model.state_dict(), path + "trained/policy_features.pt")
        torch.save(self.mu_model.state_dict(), path + "trained/policy_mu.pt")
        torch.save(self.log_std_model.state_dict(), path + "trained/policy_log_std.pt")

    def load(self, path):
        self.features_model.load_state_dict(torch.load(path + "trained/policy_features.pt", map_location = self.device))
        self.mu_model.load_state_dict(torch.load(path + "trained/policy_mu.pt", map_location = self.device))
        self.log_std_model.load_state_dict(torch.load(path + "trained/policy_log_std.pt", map_location = self.device))
