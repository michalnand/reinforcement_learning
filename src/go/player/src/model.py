import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResidualBlock(torch.nn.Module):
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()

        self.layers = [ 
                        nn.BatchNorm2d(input_channels),
                        nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(), 
                        nn.BatchNorm2d(input_channels),
                        nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()
                    ]

        self.model = nn.Sequential(*self.layers)

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)
        
    def forward(self, x):
        return x + self.model(x)

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        fc_input_height = self.input_shape[1]
        fc_input_width  = self.input_shape[2]    

        kernels_count   = 64
        fc_inputs_count = fc_input_width*fc_input_height*kernels_count
 
        self.layers = [ 
                        nn.Conv2d(input_channels, kernels_count, kernel_size=1, stride=1, padding=0),
                        nn.ReLU(), 
                        ResidualBlock(kernels_count),
                        ResidualBlock(kernels_count),

                        nn.Conv2d(kernels_count, kernels_count, kernel_size=1, stride=1, padding=0),
                        nn.ReLU(), 
                        ResidualBlock(kernels_count),
                        ResidualBlock(kernels_count),

                        nn.Conv2d(kernels_count, kernels_count, kernel_size=1, stride=1, padding=0),
                        nn.ReLU(), 
                        ResidualBlock(kernels_count),
                        ResidualBlock(kernels_count),

                        nn.Conv2d(kernels_count, kernels_count, kernel_size=1, stride=1, padding=0),
                        nn.ReLU(), 
                        ResidualBlock(kernels_count),
                        ResidualBlock(kernels_count),

                        Flatten(),             
                        nn.Linear(fc_inputs_count, outputs_count)
                    ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)

    def forward(self, state):
        return self.model.forward(state)

    def get_q_values(self, state):
        with torch.no_grad():
            state_dev       = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
            network_output  = self.model.forward(state_dev)

            return network_output[0].to("cpu").detach().numpy()
    
    def save(self, path):
        name = path + "trained/model.pt"
        print("saving", name)
        torch.save(self.model.state_dict(), name)

    def load(self, path):
        name = path + "trained/model.pt"
        print("loading", name)

        self.model.load_state_dict(torch.load(name))
        self.model.eval() 