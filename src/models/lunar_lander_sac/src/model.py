import torch
import torch.nn as nn


class FeaturesNetwork(torch.nn.Module):
    def __init__(self, input_shape, features_count = 256):
        super(FeaturesNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [ 
                        nn.Linear(input_shape[0], features_count),
                        nn.ReLU()                     
                    ]

        torch.nn.init.xavier_uniform_(self.layers[0].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

    def forward(self, state)
        return self.model(state) 

    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/features.pt")
      
    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "trained/features.pt", map_location = self.device))
  
   
class Value(torch.nn.Module):
    def __init__(self, features_count, neurons_count = 64):
        super(Value, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [ 
                            nn.Linear(features_count, neurons_count),
                            nn.ReLU(),     
                            nn.Linear(neurons_count, 1)                 
                        ]

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

    def forward(self, features):
        return self.model(features)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/value.pt")
      
    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "trained/value.pt", map_location = self.device))


class SoftQ(torch.nn.Module):
    def __init__(self, features_count, actions_count, neurons_count = 64):
        super(SoftQ, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [ 
                            nn.Linear(features_count + actions_count, neurons_count),
                            nn.ReLU(),     
                            nn.Linear(neurons_count, 1)                 
                        ]

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

    def forward(self, features):
        return self.model(features)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/soft_q.pt")
      
    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "trained/soft_q.pt", map_location = self.device))
       


class Policy(torch.nn.Module):
    def __init__(self, inputs_count, actions_count, neurons_count = 64):
        super(SoftQ, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.features_layers = [ 
                                    nn.Linear(inputs_count, neurons_count),
                                    nn.ReLU()                      
                                ]

        self.mu_layers = [
                            nn.Linear(neurons_count, actions_count),
                            nn.Tanh()
                        ]

        self.var_layers = [
                            nn.Linear(neurons_count, actions_count),
                            nn.Softplus() 
                        ]


        torch.nn.init.xavier_uniform_(self.features_layers[0].weight)
        torch.nn.init.xavier_uniform_(self.mu_layers[0].weight)
        torch.nn.init.xavier_uniform_(self.var_layers[0].weight)

        self.features_model = nn.Sequential(*self.features_layers)
        self.features_model.to(self.device)
        
        self.mu_model = nn.Sequential(*self.mu_layers)
        self.mu_model.to(self.device)

        self.var_model = nn.Sequential(*self.var_layers)
        self.var_model.to(self.device)


    def forward(self, features):
        f = self.features_model(features)
        return self.mu_model(f), self.var_model(f)
    
    def save(self, path):
        torch.save(self.features_model.state_dict(), path + "trained/policy_features.pt")
        torch.save(self.mu_model.state_dict(), path + "trained/policy_mu.pt")
        torch.save(self.var_model.state_dict(), path + "trained/policy_var.pt")

    def load(self, path):
        self.features_model.load_state_dict(torch.load(path + "trained/policy_features.pt", map_location = self.device))
        self.mu_model.load_state_dict(torch.load(path + "trained/policy_mu.pt", map_location = self.device))
        self.var_model.load_state_dict(torch.load(path + "trained/policy_var.pt", map_location = self.device))



    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample()
        action = torch.tanh(mean+ std*z.to(device))
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to(device)
        action = torch.tanh(mean + std*z)
        
        action  = action.cpu()
        return action[0]