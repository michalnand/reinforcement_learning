import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels   = self.input_shape[0]
        input_width      = self.input_shape[1]

       
        '''
        self.layers = [
                        nn.Conv1d(input_channels, 8, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(), 

                        nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(), 

                        nn.MaxPool1d(kernel_size=1, stride=2, padding=0),
 
                        Flatten(),  
                        
                        nn.Linear(8*input_width//2, 64),
                        nn.ReLU(), 
                         
                        nn.Linear(64, outputs_count)
                    ] 

        '''
        self.layers = [        
                        Flatten(),  
                        
                        nn.Linear(input_channels*input_width, 64),
                        nn.ReLU(), 
                        
                        nn.Linear(64, 32),
                        nn.ReLU(), 
                         
                        nn.Linear(32, outputs_count)
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