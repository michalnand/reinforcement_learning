import torch
import torch.nn as nn
import torchviz

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("computing device ", self.device)

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        fc_input_height = self.input_shape[1]
        fc_input_width  = self.input_shape[2]
       
        ratio           = 2**4

        fc_inputs_count = ((fc_input_width)//ratio)*((fc_input_height)//ratio)

        input_channels = self.input_shape[0]

        self.layers_features = [ 
                            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32), 
                            nn.ReLU(), 
                            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32), 
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64), 
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64), 
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64), 
                            nn.ReLU(),

                            Flatten()
                        ]

        self.layers_value = [
                            nn.Linear(fc_inputs_count*64, 512),
                            nn.ReLU(),                      
                            nn.Linear(512, 1)
                        ]

        self.layers_advantage = [
                                nn.Linear(fc_inputs_count*64, 512),
                                nn.ReLU(),                      
                                nn.Linear(512, outputs_count)
                            ]
  
        for i in range(len(self.layers_features)):
            if isinstance(self.layers_features[i], nn.Conv2d):
                torch.nn.init.xavier_uniform(self.layers_features[i].weight)

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_value = nn.Sequential(*self.layers_value)
        self.model_value.to(self.device)

        self.model_advantage = nn.Sequential(*self.layers_advantage)
        self.model_advantage.to(self.device)



    def forward(self, state):
        features    = self.model_features(state)
        value       = self.model_value(features)
        advantage   = self.model_advantage(features)

        result = value + advantage - advantage.mean()
        return result

    def get_q_values(self, state):
        with torch.no_grad():
            state_dev       = torch.tensor(state, dtype=torch.float32).detach().to(self.device)
            network_output  = self.forward(state_dev)

            return network_output[0].to("cpu").detach().numpy()
    
    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "trained/model_features.pt")
        torch.save(self.model_value.state_dict(), path + "trained/model_value.pt")
        torch.save(self.model_advantage.state_dict(), path + "trained/model_advantage.pt")

        self.render(path)

    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "trained/model_features.pt"))
        self.model_value.load_state_dict(torch.load(path + "trained/model_value.pt"))
        self.model_advantage.load_state_dict(torch.load(path + "trained/model_advantage.pt"))
        
        self.model_features.eval() 
        self.model_value.eval() 
        self.model_advantage.eval() 

    def render(self, path):

        print("rendering ", path)

        x = torch.zeros(1, self.input_shape[0], self.input_shape[1], self.input_shape[2], dtype=torch.float, requires_grad=False).to(self.device)
        out = self.forward(x)
        dot = torchviz.make_dot(out)
         
        dot.format = "svg"
        dot.render(path + "trained/model")
    