import torch
import torch.nn as nn
import numpy

class Flatten(nn.Module):
    def forward(self, input):
        #return input.view(input.size(0), -1)
        return input.view(-1, input.size(0))

class Model(torch.nn.Module):

    def __init__(self, input_shape, actions_count):
        super(Model, self).__init__()

        self.device = "cpu" 
        self.actions_count = actions_count
        
        features_outputs_count = 128
        inverse_inputs_count   = features_outputs_count*2
        forward_inputs_count   = features_outputs_count + actions_count
 

        self.layers_features = [
                                    nn.Linear(input_shape[0], features_outputs_count),
                                    nn.ReLU() 
                                ]


        self.layers_q_values  = [
                                    nn.Linear(features_outputs_count, 64),
                                    nn.ReLU(),                      
                                    nn.Linear(64, actions_count) 
                                ] 

        self.layers_inverse = [
                                nn.Linear(inverse_inputs_count, actions_count*4),
                                nn.ReLU(),                      
                                nn.Linear(actions_count*4, actions_count)
                            ] 

        self.layers_forward = [
                                nn.Linear(forward_inputs_count, forward_inputs_count),
                                nn.ReLU(),                      
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
            state_dev   = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
            action      = torch.zeros(self.actions_count,  dtype=torch.float32).to(self.device).unsqueeze(0)

            q_values, _, _, = self.forward(state_dev, state_dev, action)

            return q_values[0].to("cpu").detach().numpy() 


    def forward(self, state_now, state_next, action):
        features_now  = self.model_features.forward(state_now)
        features_next = self.model_features.forward(state_next)
        
        q_values                = self.model_q_values.forward(features_now)
        action_predicted        = self.model_inverse.forward(torch.cat((features_now, features_next), dim = 1))
        features_next_predicted = self.model_forward.forward(torch.cat((features_now, action), dim = 1))

        curiosity = ((features_next - features_next_predicted)**2).mean(dim=1)

        return q_values, curiosity, action_predicted
    