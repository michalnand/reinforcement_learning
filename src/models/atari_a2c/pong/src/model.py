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
        
        input_channels  = self.input_shape[0]
        fc_input_height = self.input_shape[1]
        fc_input_width  = self.input_shape[2]    

        ratio           = 2**4

        fc_inputs_count = ((fc_input_width)//ratio)*((fc_input_height)//ratio)
 
        self.layers_features = [ 
                                nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(), 
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        
                                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                
                                Flatten()
                            ]

        self.layers_policy = [
                                nn.Linear(fc_inputs_count*64, 512),
                                nn.ReLU(),                      
                                nn.Linear(512, outputs_count)
                            ]

        self.layers_value = [
                                nn.Linear(fc_inputs_count*64, 512),
                                nn.ReLU(),                      
                                nn.Linear(512, 1)
                            ]

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_policy   = nn.Sequential(*self.layers_policy)
        self.model_value    = nn.Sequential(*self.layers_value)

        self.model_features.to(self.device)
        self.model_policy.to(self.device)
        self.model_value.to(self.device)

        print(self.model_features)
        print(self.model_policy)
        print(self.model_value)

    def forward(self, state):
        features = self.model_features.forward(state)
        return self.model_policy.forward(features), self.model_value.forward(features)

    def get_policy(self, state):
        with torch.no_grad():
            state_dev  = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
            policy, _  = self.forward(state_dev)

            return policy[0].to("cpu").detach().numpy()
    
    def save(self, path):
        print("saving to ", path)

        torch.save(self.model_features.state_dict(), path + "trained/model_features.pt")
        torch.save(self.model_policy.state_dict(), path + "trained/model_policy.pt")
        torch.save(self.model_value.state_dict(), path + "trained/model_value.pt")

    def load(self, path):
        
        print("loading from ", path)

        self.model_features.load_state_dict(torch.load(path + "trained/model_features.pt"))
        self.model_features.eval() 

        self.model_policy.load_state_dict(torch.load(path + "trained/model_policy.pt"))
        self.model_policy.eval() 

        self.model_value.load_state_dict(torch.load(path + "trained/model_value.pt"))
        self.model_value.eval()  
    