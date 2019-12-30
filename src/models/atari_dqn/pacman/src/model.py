import torch
import torch.nn as nn

import numpy

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).cuda()
    def close(self):
        self.hook.remove()

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

    
    def kernel_visualise(self, layer, kernel, iterations = 1000):
        input_initial   = 0.01*(2.0*torch.rand((1, ) + self.input_shape, device = self.device, requires_grad=True) - 1.0)

        
        input_var       = torch.autograd.Variable(input_initial, requires_grad=True) 


        optimizer = torch.optim.Adam([input_var], lr=0.01, weight_decay=0.0000001)

        for _ in range(iterations):
            x = input_var

            optimizer.zero_grad()

            for l in range(layer+2):
                x = self.layers[l].forward(x)

            loss = -2.0*x[0][kernel].mean() + x[0].mean()
            loss.backward()
            optimizer.step()

        input_var   = input_var.squeeze(0)
        result      = input_var.to("cpu").detach().numpy()

        max = result.max()
        min = result.min()

        k = 1.0/(max - min)
        q = 1.0 - k*max

        result = result*k + q

        result = numpy.clip(result, 0.0, 1.0)
 
        return result
