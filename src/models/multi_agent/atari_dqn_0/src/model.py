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
 
        self.layers = [ 
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
                        
                        Flatten(), 
                        nn.Linear(fc_inputs_count*64, 512),
                        nn.ReLU(),                      

                        nn.Linear(512, outputs_count)
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
     

    def get_activity_map(self, state):
        with torch.no_grad():
            x  = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)

            for i in range(12):
                x = self.layers[i].forward(x)

            upsample = nn.Upsample(size=(self.input_shape[1], self.input_shape[2]), mode='bicubic')

            x = upsample(x)
            x = x.sum(dim = 1)
            result = x[0].to("cpu").detach().numpy()

            k = 1.0/(result.max() - result.min())
            q = 1.0 - k*result.max()
            result = k*result + q
            
            return result