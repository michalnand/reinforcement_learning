import torch
import torch.nn as nn

class ModelPolicy(torch.nn.Module):
    def __init__(self, inputs_count, hidden_count, outputs_count):
        super(ModelPolicy, self).__init__()

        self.layers = [
            torch.nn.Linear(inputs_count, hidden_count),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_count, outputs_count)
        ] 

        torch.nn.init.orthogonal_(self.layers[0].weight, 0.01)
        torch.nn.init.zeros_(self.layers[0].bias)

        torch.nn.init.orthogonal_(self.layers[2].weight, 0.01)
        torch.nn.init.zeros_(self.layers[2].bias)

        self.model = nn.Sequential(*self.layers)
     
    def forward(self, x):
        return self.model(x)
     

class ModelCritic(torch.nn.Module): 
    def __init__(self, inputs_count, hidden_count):
        super(ModelCritic, self).__init__()

        self.hidden     = torch.nn.Linear(inputs_count, hidden_count)
        self.act        = torch.nn.ReLU()

        self.ext_value  = torch.nn.Linear(hidden_count, 1)
        self.int_value  = torch.nn.Linear(hidden_count, 1)

        torch.nn.init.orthogonal_(self.hidden.weight, 0.1)
        torch.nn.init.zeros_(self.hidden.bias)

        torch.nn.init.orthogonal_(self.ext_value.weight, 0.01)
        torch.nn.init.zeros_(self.ext_value.bias)

        torch.nn.init.orthogonal_(self.int_value.weight, 0.01)
        torch.nn.init.zeros_(self.int_value.bias)

    def forward(self, x): 
        y = self.hidden(x)
        y = self.act(y)
        
        return self.ext_value(y), self.int_value(y)


class ModelSymmetry(torch.nn.Module):
    def __init__(self, in_ch, actions_count, hidden_count = 128):
        super(ModelSymmetry, self).__init__()

        self.layers_attention = [
            torch.nn.Conv2d(in_ch, hidden_count, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_count, 1, kernel_size=1, stride=1, padding=0)
        ] 

        self.layers_action = [
            torch.nn.Conv2d(2*in_ch, hidden_count, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_count, actions_count, kernel_size=1, stride=1, padding=0)
        ]

        self.layers_symmetry = [
            torch.nn.Conv2d(2*in_ch, hidden_count, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_count, hidden_count, kernel_size=1, stride=1, padding=0)
        ]
 
        torch.nn.init.orthogonal_(self.layers_attention[0].weight, 0.01)
        torch.nn.init.zeros_(self.layers_attention[0].bias)
        torch.nn.init.orthogonal_(self.layers_attention[2].weight, 0.01)
        torch.nn.init.zeros_(self.layers_attention[2].bias)

        torch.nn.init.orthogonal_(self.layers_action[0].weight, 0.01)
        torch.nn.init.zeros_(self.layers_action[0].bias)
        torch.nn.init.orthogonal_(self.layers_action[2].weight, 0.01)
        torch.nn.init.zeros_(self.layers_action[2].bias)

        torch.nn.init.orthogonal_(self.layers_symmetry[0].weight, 0.01)
        torch.nn.init.zeros_(self.layers_symmetry[0].bias)
        torch.nn.init.orthogonal_(self.layers_symmetry[2].weight, 0.01)
        torch.nn.init.zeros_(self.layers_symmetry[2].bias) 


        self.model_attention    = nn.Sequential(*self.layers_attention)
        self.model_action       = nn.Sequential(*self.layers_action)
        self.model_symmetry     = nn.Sequential(*self.layers_symmetry)

     
    def forward(self, features_now, features_next):
        features = torch.cat([features_now, features_next], dim=1)
        
        attn    = self.model_attention(features_now)
        attn    = torch.sigmoid(attn)

        action  = self.model_action(features)*attn
        action  = action.sum(dim=(2, 3))

        features= self.model_symmetry(features)*attn
        features= features.sum(dim=(2, 3))

        return features, action
 
class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    

        features_count  = 128*(input_height//8)*(input_width//8)
        hidden_count    = 512   
        
  
        self.layers_features = [   
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(features_count, hidden_count),
            nn.ReLU()
        ]     

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers_features[i].bias)

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device) 

        self.model_policy = ModelPolicy(hidden_count, hidden_count, outputs_count)
        self.model_policy.to(self.device)

        self.model_critic = ModelCritic(hidden_count, hidden_count)
        self.model_critic.to(self.device) 


        print("model_ppo")
        print(self.model_features)
        print(self.model_policy)
        print(self.model_critic)
        print("\n\n")

    def forward(self, state):
        features                = self.model_features(state)
        policy                  = self.model_policy(features)
        ext_value, int_value    = self.model_critic(features)

        return policy, ext_value, int_value

       
    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "model_features.pt")
        torch.save(self.model_policy.state_dict(), path + "model_policy.pt")
        torch.save(self.model_critic.state_dict(), path + "model_critic.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "model_features.pt", map_location = self.device))
        self.model_policy.load_state_dict(torch.load(path + "model_policy.pt", map_location = self.device))
        self.model_critic.load_state_dict(torch.load(path + "model_critic.pt", map_location = self.device))

        self.model_features.eval()  
        self.model_policy.eval() 
        self.model_critic.eval()


if __name__ == "__main__":
    state_shape     = (128, 12, 12)
    actions_count   = 18
    batch_size      = 32


    model = ModelSymmetry(128, actions_count)

    print(model)

    state           = torch.randn((batch_size, ) + state_shape)

    features, action = model(state, state)

    print(">>> ", features.shape, action.shape)


    '''
    model           = Model(state_shape, actions_count)

    state           = torch.randn((batch_size, ) + state_shape)
    
    policy, ext_value, int_value = model(state)

    print("shape = ", policy.shape, ext_value.shape, int_value.shape)
    '''