import torch
import torch.nn as nn
import torchviz

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [ 
                        nn.BatchNorm2d(input_channels),
                        nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(), 
                        nn.BatchNorm2d(input_channels),
                        nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()
                    ]

        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return x + self.model(x)

class NoiseLayer(torch.nn.Module):
    def __init__(self, inputs_count, init_range = 0.001):
        super(NoiseLayer, self).__init__()
        
        self.inputs_count   = inputs_count
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        w_initial   = init_range*(2.0*torch.rand(self.inputs_count, device = self.device) - 1.0)
        self.w      = torch.nn.Parameter(w_initial, requires_grad = True)     

    def forward(self, x):
        noise = (torch.rand(self.inputs_count, device = self.device)*2.0 - 1.0).detach()
        return x + self.w*noise



class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    


        layer_0_kernels_count = 32
        layer_1_kernels_count = 32
        layer_2_kernels_count = 64
        layer_3_kernels_count = 64

        scale_ratio           = 2**4

        fc_inputs_count = ((input_width)//scale_ratio)*((input_height)//scale_ratio)*layer_3_kernels_count
        
        self.layers_features = [ 
                                nn.Conv2d(input_channels, layer_0_kernels_count, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(), 
                                ResidualBlock(layer_0_kernels_count),
                                ResidualBlock(layer_0_kernels_count),

                                nn.Conv2d(layer_0_kernels_count, layer_1_kernels_count, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(), 
                                ResidualBlock(layer_1_kernels_count),
                                ResidualBlock(layer_1_kernels_count),

                                nn.Conv2d(layer_1_kernels_count, layer_2_kernels_count, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(), 
                                ResidualBlock(layer_2_kernels_count),
                                ResidualBlock(layer_2_kernels_count),
                                ResidualBlock(layer_2_kernels_count),
                                ResidualBlock(layer_2_kernels_count),

                                nn.Conv2d(layer_2_kernels_count, layer_3_kernels_count, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(), 
                                ResidualBlock(layer_3_kernels_count),
                                ResidualBlock(layer_3_kernels_count),
                                ResidualBlock(layer_3_kernels_count),
                                ResidualBlock(layer_3_kernels_count),

                                Flatten(),
                                NoiseLayer(fc_inputs_count, 0.001)
                            ]
                            
        self.layers_value = [
                            nn.Linear(fc_inputs_count, 512),
                            nn.ReLU(),                      
                            nn.Linear(512, 1)
                        ]

        self.layers_advantage = [
                                nn.Linear(fc_inputs_count, 512),
                                nn.ReLU(),                      
                                nn.Linear(512, outputs_count)
                            ]
  
        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)


        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_value = nn.Sequential(*self.layers_value)
        self.model_value.to(self.device)

        self.model_advantage = nn.Sequential(*self.layers_advantage)
        self.model_advantage.to(self.device)

        print(self.model_features)
        print(self.model_value)
        print(self.model_advantage)



    def forward(self, state):
        features    = self.model_features(state)
        value       = self.model_value(features)
        advantage   = self.model_advantage(features)

        result = value + advantage - advantage.mean()
        return result

    def get_q_values(self, state):
        with torch.no_grad():
            state_dev       = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
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

    def render(self, path = "./"):

        print("rendering ", path)

        x = torch.zeros(32, self.input_shape[0], self.input_shape[1], self.input_shape[2], dtype=torch.float, requires_grad=False).to(self.device)
        out = self.forward(x)
        dot = torchviz.make_dot(out)
         
        dot.format = "svg"
        dot.render(path + "trained/model")
        #dot.render(path + "model")
    
