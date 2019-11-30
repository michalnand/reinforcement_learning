import torch
import torch.nn as nn
import numpy

class Flatten(nn.Module):
    def forward(self, input):
        #return input.view(input.size(0), -1)
        return input.view(-1, input.size(0))

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [ 
                       
                        nn.Linear(input_shape[0], 64),
                        nn.ReLU(), 

                        nn.Linear(64, 64),
                        nn.ReLU(), 

                        nn.Linear(64, outputs_count)
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
            rs = numpy.reshape(state, (1, ) + state.shape)

            state_dev       = torch.tensor(rs, dtype=torch.float32).detach().to(self.device)
            network_output  = self.model.forward(state_dev)

            return network_output[0].to("cpu").detach().numpy()

    