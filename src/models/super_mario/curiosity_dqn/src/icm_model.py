import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):

    def __init__(self, input_shape, actions_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_channels  = input_shape[0]
        fc_input_height = input_shape[1]
        fc_input_width  = input_shape[2]    

        ratio           = 2**4

        features_outputs_count = 64*((fc_input_width)//ratio)*((fc_input_height)//ratio)
 
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

        self.layers_inverse = [
                                nn.Linear(features_outputs_count*2, 256),
                                nn.ReLU(),                      
                                nn.Linear(256, actions_count) 
                            ]

        self.layers_forward = [
                                nn.Linear(features_outputs_count + actions_count, 1024),
                                nn.ReLU(),                      
                                nn.Linear(1024, features_outputs_count) 
                            ]

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)


        self.model_features = nn.Sequential(*self.layers_features)
        self.model_inverse = nn.Sequential(*self.layers_inverse)
        self.model_forward = nn.Sequential(*self.layers_forward)

        self.model_features.to(self.device)
        self.model_inverse.to(self.device)
        self.model_forward.to(self.device)

        print(self.model_features)
        print(self.model_inverse)
        print(self.model_forward)

    def forward(self, state_now, state_next, action):
        features_now  = self.model_features.forward(state_now)
        features_next = self.model_features.forward(state_next)
        
        action_predicted        = self.model_inverse.forward(torch.cat((features_now, features_next), dim = 1))
        features_next_predicted = self.model_forward.forward(torch.cat((features_now, action), dim = 1))

        curiosity = ((features_next - features_next_predicted)**2).sum(dim=1)

        return curiosity, action_predicted
    
    def save(self, path):
        torch.save(self.model_features.state_dict(), path + "trained/icm_model_features.pt")
        torch.save(self.model_inverse.state_dict(), path + "trained/icm_model_inverse.pt")
        torch.save(self.model_forward.state_dict(), path + "trained/icm_model_forward.pt")

    def load(self, path):
        name = path + "trained/model.pt"
        print("loading", name)

        self.model.load_state_dict(torch.load(name))
        self.model.eval() 
     
