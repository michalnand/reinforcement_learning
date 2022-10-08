import torch
import torch.nn as nn

class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, sigma = 1.0):
        super(NoisyLinear, self).__init__()
        
        self.out_features   = out_features
        self.in_features    = in_features
        self.sigma          = sigma

        self.weight  = nn.Parameter(torch.zeros(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias  = nn.Parameter(torch.zeros(out_features))
 

        self.weight_noise  = nn.Parameter(torch.zeros(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight_noise)

        self.bias_noise  = nn.Parameter((0.1/out_features)*torch.randn(out_features)) 
 

    def forward(self, x): 
        col_noise       = torch.randn((1, self.out_features)).to(x.device).detach()
        row_noise       = torch.randn((self.in_features, 1)).to(x.device).detach()

        weight_noise    = self.sigma*row_noise.matmul(col_noise)

        bias_noise      = self.sigma*torch.randn((self.out_features)).to(x.device).detach()

        weight_noised   = self.weight + self.weight_noise*weight_noise
        bias_noised     = self.bias   + self.bias_noise*bias_noise 

        return x.matmul(weight_noised) + bias_noised

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    


        fc_inputs_count = 128*(input_width//16)*(input_height//16)
  
        self.layers_features = [ 
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten()
        ] 

        self.layers_value = [
            nn.Linear(fc_inputs_count, 512),
            nn.ReLU(),                       
            nn.Linear(512, 1)    
        ]  

        self.layers_advantage = [
            NoisyLinear(fc_inputs_count, 512),
            nn.ReLU(),                      
            NoisyLinear(512, outputs_count)
        ]
 
  
        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)

        for i in range(len(self.layers_value)):
            if hasattr(self.layers_value[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_value[i].weight)

        for i in range(len(self.layers_advantage)):
            if hasattr(self.layers_advantage[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_advantage[i].weight)


        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_value = nn.Sequential(*self.layers_value)
        self.model_value.to(self.device)

        self.model_advantage = nn.Sequential(*self.layers_advantage)
        self.model_advantage.to(self.device)

        print("model_dqn")
        print(self.model_features)
        print(self.model_value)
        print(self.model_advantage)
        print("\n\n")


    def forward(self, state):
        features    = self.model_features(state)

        value       = self.model_value(features)
        advantage   = self.model_advantage(features)

        result = value + advantage - advantage.mean(dim=1, keepdim=True)

        return result

    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "model_features.pt")
        torch.save(self.model_value.state_dict(), path + "model_value.pt")
        torch.save(self.model_advantage.state_dict(), path + "model_advantage.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "model_features.pt", map_location = self.device))
        self.model_value.load_state_dict(torch.load(path + "model_value.pt", map_location = self.device))
        self.model_advantage.load_state_dict(torch.load(path + "model_advantage.pt", map_location = self.device))
        
        self.model_features.eval() 
        self.model_value.eval() 
        self.model_advantage.eval() 


    def get_activity_map(self, state):
 
        state_t     = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
        features    = self.model_features(state_t)
        features    = features.reshape((1, 128, 6, 6))

        upsample = nn.Upsample(size=(self.input_shape[1], self.input_shape[2]), mode='bicubic')

        features = upsample(features).sum(dim = 1)

        result = features[0].to("cpu").detach().numpy()

        k = 1.0/(result.max() - result.min())
        q = 1.0 - k*result.max()
        result = k*result + q
        
        return result


if __name__ == "__main__":
    batch_size = 8

    channels = 4
    height   = 96
    width    = 96

    actions_count = 9


    state   = torch.rand((batch_size, channels, height, width))

    model = Model((channels, height, width), actions_count)


    q_values = model.forward(state)

    print(q_values.shape)




