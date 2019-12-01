import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        
        neurons_count = 64
 
        self.features_layers = [ 
                                    nn.Linear(input_shape[0], neurons_count),
                                    nn.ReLU(),                      
                                    nn.Linear(neurons_count, neurons_count),
                                    nn.ReLU(),
                            ]

        self.layers_policy = [
                                nn.Linear(neurons_count, neurons_count),
                                nn.ReLU(),                      
                                nn.Linear(neurons_count, outputs_count),
                                nn.Softmax()
                            ]

        self.layers_critic = [ 
                                nn.Linear(neurons_count, neurons_count),
                                nn.ReLU(),                      
                                nn.Linear(neurons_count, 1)
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


        print(self.model_features)
        print(self.model_policy)
        print(self.model_critic)


    def forward(self, state):
        features_output =  self.model_features(state)

        policy_output = self.model_policy(features_output)
        critic_output = self.model_critic(features_output)

      
        return policy_output, critic_output


    def get_output(self, state):
        with torch.no_grad():
            state_dev = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
            policy_output, critic_output  = self.forward(state_dev)

            return policy_output[0].to("cpu").detach().numpy(), critic_output[0].to("cpu").detach().numpy()
    
    def save(self, path):
        print("saving to ", path)

        torch.save(self.model_features.state_dict(), path + "trained/model_features.pt")
        torch.save(self.model_policy.state_dict(), path + "trained/model_policy.pt")
        torch.save(self.model_critic.state_dict(), path + "trained/model_critic.pt")

    def load(self, path):
        
        print("loading from ", path)

        self.model_features.load_state_dict(torch.load(path + "trained/model_features.pt"))
        self.model_features.eval() 

        self.model_policy.load_state_dict(torch.load(path + "trained/model_policy.pt"))
        self.model_policy.eval() 

        self.model_critic.load_state_dict(torch.load(path + "trained/model_critic.pt"))
        self.model_critic.eval()  
    