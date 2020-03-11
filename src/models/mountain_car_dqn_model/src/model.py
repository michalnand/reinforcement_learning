import torch
import torch.nn as nn
import numpy

class Flatten(nn.Module):
    def forward(self, input):
        #return input.view(input.size(0), -1)
        return input.view(-1, input.size(0))

class ModelDQNICM(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(ModelDQNICM, self).__init__()

        self.device = "cpu"

        self.output_count = outputs_count

        features_count = 64

        self.layers_features = [      
                                    nn.Linear(input_shape[0], features_count),
                                    nn.ReLU()
                                ]

        self.layers_inverse = [
                                nn.Linear(features_count*2, 64),
                                nn.ReLU(), 

                                nn.Linear(64, outputs_count)
                            ]

        self.layers_forward = [
                                nn.Linear(features_count + outputs_count, 64),
                                nn.ReLU(), 

                                nn.Linear(64, features_count)
                            ]

        self.layers_dqn = [
                            nn.Linear(features_count, 64),
                            nn.ReLU(), 

                            nn.Linear(64, outputs_count)
                        ]


        self.model_features = nn.Sequential(*self.layers_features)
        self.model_inverse  = nn.Sequential(*self.layers_inverse)
        self.model_forward  = nn.Sequential(*self.layers_forward)
        self.model_dqn      = nn.Sequential(*self.layers_dqn)

        self.model_features.to(self.device)
        self.model_inverse.to(self.device)
        self.model_forward.to(self.device)
        self.model_dqn.to(self.device)

        print(self.model_features)
        print(self.model_inverse)
        print(self.model_forward)
        print(self.model_dqn)


    def forward(self, state, state_next, action):
        f_now  = self.model_features(state)
        f_next = self.model_features(state_next)

        q_values = self.model_dqn(f_now)

        inverse_input           = torch.cat([f_now, f_next], dim = 1)
        action_predicted        = self.model_inverse(inverse_input)


        forward_input           = torch.cat([f_now, action], dim = 1)
        features_predicted      = self.model_forward(forward_input)

        curiosity = ((f_next - features_predicted)**2.0).mean(dim = 1)

        return q_values, action_predicted, curiosity


    def get_q_values(self, state):
        with torch.no_grad():
            rs = numpy.reshape(state, (1, ) + state.shape)

            action_dummy = torch.zeros((1, self.output_count),  dtype=torch.float32).to(self.device)


            state_dev       = torch.tensor(rs, dtype=torch.float32).detach().to(self.device)
            network_output, _, _  = self.forward(state_dev, state_dev, action_dummy)

            return network_output[0].to("cpu").detach().numpy()

    