import torch
import torch.nn as nn

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    

        features_count  = 64*(input_width//8)*(input_height//8)

        hidden_count    = 512  
  
        self.layers_features = [   
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Flatten()
        ]    

        self.layers_value = [
            nn.Linear(features_count, hidden_count),
            nn.ReLU(),                       
            nn.Linear(hidden_count, 1)    
        ]  

        self.layers_policy = [
            nn.Linear(features_count, hidden_count),
            nn.ReLU(),                      
            nn.Linear(hidden_count, outputs_count)
        ]

        self.layers_predictor = [
            nn.Linear(2*features_count, hidden_count),
            nn.ReLU(),                      
            nn.Linear(hidden_count, hidden_count)
        ]

        #init features
        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers_features[i].bias)

        #init critic
        torch.nn.init.orthogonal_(self.layers_value[0].weight, 0.1)
        torch.nn.init.zeros_(self.layers_value[0].bias)
        torch.nn.init.orthogonal_(self.layers_value[2].weight, 0.01)
        torch.nn.init.zeros_(self.layers_value[2].bias)

        #init actor
        for i in range(len(self.layers_policy)):
            if hasattr(self.layers_policy[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_policy[i].weight, 0.01)
                torch.nn.init.zeros_(self.layers_policy[i].bias)


        #init predictor
        for i in range(len(self.layers_predictor)):
            if hasattr(self.layers_predictor[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_predictor[i].weight, 0.1)
                torch.nn.init.zeros_(self.layers_predictor[i].bias)


        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_value = nn.Sequential(*self.layers_value)
        self.model_value.to(self.device)

        self.model_policy = nn.Sequential(*self.layers_policy)
        self.model_policy.to(self.device)

        self.model_predictor = nn.Sequential(*self.layers_predictor)
        self.model_predictor.to(self.device)

        print("model_ppo")
        print(self.model_features)
        print(self.model_value)
        print(self.model_policy)
        print(self.model_predictor)
        print("\n\n")


    def forward(self, state):
        features    = self.model_features(state)

        value    = self.model_value(features)
        policy   = self.model_policy(features)

        return policy, value

    def forward_features(self, state_now, state_next):
        features_now  = self.model_features(state_now)
        features_next = self.model_features(state_next)

        features = torch.cat([features_now, features_next], dim=1)

        return self.model_predictor(features)
        
    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "model_features.pt")
        torch.save(self.model_value.state_dict(), path + "model_value.pt")
        torch.save(self.model_policy.state_dict(), path + "model_policy.pt")
        torch.save(self.model_predictor.state_dict(), path + "model_predictor.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "model_features.pt", map_location = self.device))
        self.model_value.load_state_dict(torch.load(path + "model_value.pt", map_location = self.device))
        self.model_policy.load_state_dict(torch.load(path + "model_policy.pt", map_location = self.device))
        self.model_predictor.load_state_dict(torch.load(path + "model_predictor.pt", map_location = self.device))

