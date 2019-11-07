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

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count


        print("computing device ", self.device)
        print("input_shape ", self.input_shape)
        print("outputs_count ", self.outputs_count)

        
        fc_input_height = self.input_shape[2]
        fc_input_width  = self.input_shape[3]
       

        ratio           = 2**3

        fc_inputs_count = ((fc_input_width)//ratio)*((fc_input_height)//ratio)

        input_channels = self.input_shape[1]

        self.layers = [ 
                        nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(), 
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
 
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
 
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                        Flatten(),  
                        nn.Linear(fc_inputs_count*32, 256),
                        nn.ReLU(),                      

                        nn.Linear(256, outputs_count)
                    ]
  

        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Conv2d) or isinstance(self.layers[i], nn.Linear):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)
 
        self.model = nn.Sequential(*self.layers) 
        self.model.to(self.device)

        print(self.model)


    def forward(self, state):
        return self.model.forward(state)

    def get_q_values(self, state):
        with torch.no_grad():
            state_dev       = torch.tensor(state, dtype=torch.float32).detach().to(self.device)
            network_output  = self.model.forward(state_dev)

            q_values = network_output[0].to("cpu").detach().numpy()

            return q_values
    
    def save(self, path):
        name = path + "trained/model.pt"
        print("saving", name)
        torch.save(self.model.state_dict(), name)
 
        self.render(path)

    def load(self, path):
        name = path + "trained/model.pt"
        print("loading", name)

        self.model.load_state_dict(torch.load(name))
        self.model.eval() 
    

    def render(self, path):

        print("rendering ", path)

        x = torch.zeros(1, self.input_shape[1], self.input_shape[2], self.input_shape[3], dtype=torch.float32, requires_grad=False).to(self.device)
        out = self.forward(x)
        dot = torchviz.make_dot(out)
         
        dot.format = "svg"
        dot.render(path + "trained/model")