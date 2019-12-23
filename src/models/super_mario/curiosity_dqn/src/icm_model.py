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
        inverse_inputs_count   = features_outputs_count*2
        forward_inputs_count   = features_outputs_count + actions_count
 
        self.layers_features = [ 
                                nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
                                nn.ELU(), 
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                nn.ELU(),
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        
                                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                nn.ELU(),
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                nn.ELU(),
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                
                                Flatten()
                            ]

        self.layers_q_values  = [
                                    nn.Linear(features_outputs_count, 512),
                                    nn.ELU(),                      
                                    nn.Linear(512, actions_count) 
                                ]

        self.layers_inverse = [
                                nn.Linear(inverse_inputs_count, inverse_inputs_count//8),
                                nn.ELU(),                      
                                nn.Linear(inverse_inputs_count//8, actions_count) 
                            ]

        self.layers_forward = [
                                nn.Linear(forward_inputs_count, forward_inputs_count),
                                nn.ELU(),                      
                                nn.Linear(forward_inputs_count, features_outputs_count) 
                            ]

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)

        for i in range(len(self.layers_q_values)):
            if hasattr(self.layers_q_values[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_q_values[i].weight)

        for i in range(len(self.layers_inverse)):
            if hasattr(self.layers_inverse[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_inverse[i].weight)

        for i in range(len(self.layers_forward)):
            if hasattr(self.layers_forward[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_forward[i].weight)

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_q_values = nn.Sequential(*self.layers_q_values)
        self.model_inverse = nn.Sequential(*self.layers_inverse)
        self.model_forward = nn.Sequential(*self.layers_forward)

        self.model_features.to(self.device)
        self.model_q_values.to(self.device)
        self.model_inverse.to(self.device)
        self.model_forward.to(self.device)

        print("features model \n", self.model_features, "\n\n")
        print("q values model \n", self.model_q_values, "\n\n")
        print("inverse model \n", self.model_inverse, "\n\n")
        print("forward model \n",self.model_forward, "\n\n")


    def get_q_values(self, state):
        with torch.no_grad():
            state_dev       = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
            features_now    = self.model_features.forward(state_dev)
            q_values        = self.model_q_values.forward(features_now)

            return q_values[0].to("cpu").detach().numpy() 

    def forward(self, state_now, state_next, action):
        features_now  = self.model_features.forward(state_now)
        features_next = self.model_features.forward(state_next)
        
        q_values                = self.model_q_values.forward(features_now)
        action_predicted        = self.model_inverse.forward(torch.cat((features_now, features_next), dim = 1))
        features_next_predicted = self.model_forward.forward(torch.cat((features_now, action), dim = 1))

        curiosity = ((features_next - features_next_predicted)**2).sum(dim=1)


        return q_values, curiosity, action_predicted
    
    def save(self, path):
        torch.save(self.model_features.state_dict(), path + "trained/icm_model_features.pt")
        torch.save(self.model_q_values.state_dict(), path + "trained/icm_model_q_values.pt")
        torch.save(self.model_inverse.state_dict(), path + "trained/icm_model_inverse.pt")
        torch.save(self.model_forward.state_dict(), path + "trained/icm_model_forward.pt")

    def load(self, path):
        self.model_features.load_state_dict(torch.load(path + "trained/icm_model_features.pt"))
        self.model_features.eval() 

        self.model_q_values.load_state_dict(torch.load(path + "trained/icm_model_q_values.pt"))
        self.model_q_values.eval() 

        self.model_inverse.load_state_dict(torch.load(path + "trained/icm_model_inverse.pt"))
        self.model_inverse.eval() 
     
        self.model_forward.load_state_dict(torch.load(path + "trained/icm_model_forward.pt"))
        self.model_forward.eval() 