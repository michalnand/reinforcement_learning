import torch
import torch.nn as nn



class ValueNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim = 256, init_weight_range = 0.003):
        super(ValueNetwork, self).__init__() 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [ 
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),                     
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),    
                        nn.Linear(hidden_dim, 1)                 
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

        print(self.model)

    def forward(self, state):
        return self.model(state)  

    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/value.pt")
      
    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "trained/value.pt", map_location = self.device))


class SoftQNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 256, init_weight_range = 0.003):
        super(SoftQNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [ 
                        nn.Linear(input_dim + output_dim, hidden_dim),
                        nn.ReLU(),                     
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),    
                        nn.Linear(hidden_dim, 1)                 
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

        print(self.model)

    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1)
        return self.model(x) 

    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/soft_q.pt")
      
    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "trained/soft_q.pt", map_location = self.device))



class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 256, init_weight_range = 0.003, log_std_min = -20.0, log_std_max = 2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.features_layers = [ 
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(),                   
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU()                   
                                ]

        self.mean_layers =    [
                                nn.Linear(hidden_dim, output_dim)
                            ]

        self.log_std_layers =   [
                                    nn.Linear(hidden_dim, output_dim)
                                ]



        self.features_model = nn.Sequential(*self.features_layers)
        self.features_model.to(self.device)
        
        torch.nn.init.uniform_(self.mean_layers[0].weight, -init_weight_range, init_weight_range)
        torch.nn.init.uniform_(self.mean_layers[0].bias, -init_weight_range, init_weight_range)
        self.mean_model = nn.Sequential(*self.mean_layers)
        self.mean_model.to(self.device)

 
        torch.nn.init.uniform_(self.log_std_layers[0].weight, -init_weight_range, init_weight_range)
        torch.nn.init.uniform_(self.log_std_layers[0].bias, -init_weight_range, init_weight_range)
        self.log_std_model = nn.Sequential(*self.log_std_layers)
        self.log_std_model.to(self.device)

        print(self.features_model)
        print(self.mean_model)
        print(self.log_std_model)


    def forward(self, state):
        features = self.features_model(state)
        
        mean      = self.mean_model(features)

        log_std = self.log_std_model(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        
        std         = log_std.exp()

        normal = torch.distributions.Normal(0.0, 1.0)
        z      = normal.sample((self.output_dim, )).to(self.device)
        action = torch.tanh(mean + std*z)

        #log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = torch.distributions.Normal(mean, std).log_prob(mean + std*z) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_pi, z, mean, log_std
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        mean, log_std = self.forward(state)

        std = log_std.exp()
        
        normal = torch.distributions.Normal(0.0, 1.0)
        z      = normal.sample((self.output_dim, )).to(self.device)
        action = torch.tanh(mean + std*z)

        action  = action.detach().to("cpu").numpy()[0]
        return action

    def save(self, path):
        torch.save(self.features_model.state_dict(), path + "trained/policy_features.pt")
        torch.save(self.mean_model.state_dict(), path + "trained/policy_mean.pt")
        torch.save(self.log_std_model.state_dict(), path + "trained/policy_log_std.pt")
 
    def load(self, path):
        self.features_model.load_state_dict(torch.load(path + "trained/policy_features.pt", map_location = self.device))
        self.mean_model.load_state_dict(torch.load(path + "trained/policy_mean.pt", map_location = self.device))
        self.log_std_model.load_state_dict(torch.load(path + "trained/policy_log_std.pt", map_location = self.device))



if __name__ == "__main__":
    input_dim    = 32
    output_dim   = 7

    policy_net = PolicyNetwork(input_dim, output_dim, 256)

    state = torch.randn(input_dim).to(policy_net.device)

    action = policy_net.get_action(state)
    print("action =", action)

    print("\n\n\n")

    action, log_pi, z, mean,  log_std = policy_net.sample(state)
    print("action =", action)
    print("log_pi =", log_pi)
    print("z =", z)
    print("mean =", mean)
    print("log_std =", log_std)

   
    print("test done")