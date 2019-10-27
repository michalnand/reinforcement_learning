import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, learning_rate = 0.001):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count

        fc_input_height = self.input_shape[2]
        fc_input_width  = self.input_shape[3]
        

        ratio           = 2**4

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
            
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                        Flatten(), 
                        nn.Linear(fc_inputs_count*64, 256),
                        nn.ReLU(), 

                        nn.Linear(256, outputs_count)
                    ]

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        print(self.model)


    def forward(self, state):
        return self.model.forward(state)

    def get_q_values(self, state):
        state_dev       = torch.tensor(state, dtype=torch.float32).detach().to(self.device)
        network_output  = self.model.forward(state_dev)

        return network_output[0].to("cpu").detach().numpy()
    
    def save(self, path):
        name = path + "trained/model.pt"
        print("saving", name)
        torch.save(self.model.state_dict(), name)
    
    '''
    def train(self, input, target, loss_fn):
        self.optimizer.zero_grad()
        output = self.model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    '''