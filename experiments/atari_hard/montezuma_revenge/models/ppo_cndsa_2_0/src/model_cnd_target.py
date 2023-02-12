import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, input_shape, actions_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc_size = (input_shape[1]//8) * (input_shape[2]//8)

        self.layers = [
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
          
            nn.Flatten(),   

            nn.Linear(64*fc_size, 512)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.orthogonal_(self.layers[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers[i].bias)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)


        self.layers_action = [
            nn.Linear(2*512, 512),
            nn.ELU(),
            nn.Linear(512, actions_count)
        ]

        for i in range(len(self.layers_action)):
            if hasattr(self.layers_action[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_action[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers_action[i].bias)

        self.model_action = nn.Sequential(*self.layers_action)
        self.model_action.to(self.device)


        print("model_cnd_target")
        print(self.model)
        print(self.model_action)
        print("\n\n")

    def forward(self, state): 
        x = state[:,0,:,:].unsqueeze(1)
        return self.model(x)

    def predict_action(self, state_now, state_next):
        x_now  = state_now[:,0,:,:].unsqueeze(1)
        x_next = state_next[:,0,:,:].unsqueeze(1)
        
        z_now  = self.model(x_now)
        z_next = self.model(x_next)

        z = torch.cat([z_now, z_next], dim=1)

        return self.model_action(z)


    def save(self, path):
        torch.save(self.model.state_dict(), path + "model_cnd_target.pt")
        torch.save(self.model_action.state_dict(), path + "model_cnd_target_action.pt")

    def load(self, path):
        self.model.load_state_dict(torch.load(path + "model_cnd_target.pt", map_location = self.device))
        self.model_action.load_state_dict(torch.load(path + "model_cnd_target_action.pt", map_location = self.device))
