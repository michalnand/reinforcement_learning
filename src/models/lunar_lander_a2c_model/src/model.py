import torch
import torch.nn as nn

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu" 

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count

        self.features_count = 64
        
         
        self.features_layers = [ 
                                    nn.Linear(input_shape[0], self.features_count),
                                    nn.ReLU(),                      
                            ]

        self.layers_policy = [
                                nn.Linear(self.features_count, 32),
                                nn.ReLU(), 
                                nn.Linear(32, outputs_count)
                            ]

        self.layers_critic = [      
                                nn.Linear(self.features_count, 32),
                                nn.ReLU(),                
                                nn.Linear(32, 1)
                            ]


        self.layers_im  = [
                            nn.Linear(2*self.features_count, 32),
                            nn.ReLU(),                
                            nn.Linear(32, outputs_count)
                        ]


        for i in range(len(self.features_layers)):
            if hasattr(self.features_layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.features_layers[i].weight)


        self.model_features = nn.Sequential(*self.features_layers)
        self.model_features.to(self.device)

        self.model_policy = nn.Sequential(*self.layers_policy)
        self.model_policy.to(self.device)

        self.model_critic = nn.Sequential(*self.layers_critic)
        self.model_critic.to(self.device)

        self.model_im = nn.Sequential(*self.layers_im)
        self.model_im.to(self.device)

        print(self.model_features)
        print(self.model_policy)
        print(self.model_critic)
        print(self.model_im)


    def forward(self, state, state_prev):
        features_output      =  self.model_features(state)
        features_output_prev =  self.model_features(state_prev)

        policy_output = self.model_policy(features_output)
        critic_output = self.model_critic(features_output)

        im_input = torch.cat((features_output, features_output_prev), 1)
        inverse_model_output = self.model_im(im_input)
 
        return policy_output, critic_output, inverse_model_output
    
    def save(self, path):
        print("saving to ", path)

        torch.save(self.model_features.state_dict(), path + "trained/model_features.pt")
        torch.save(self.model_policy.state_dict(), path + "trained/model_policy.pt")
        torch.save(self.model_critic.state_dict(), path + "trained/model_critic.pt")

    def load(self, path):       
        print("loading from ", path)

        self.model_features.load_state_dict(torch.load(path + "trained/model_features.pt", map_location = self.device))
        self.model_value.load_state_dict(torch.load(path + "trained/model_policy.pt", map_location = self.device))
        self.model_advantage.load_state_dict(torch.load(path + "trained/model_critic.pt", map_location = self.device))
    
        self.model_features.eval() 
        self.model_policy.eval() 
        self.model_critic.eval()  
    