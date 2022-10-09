import torch
import torch.nn as nn




class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc_size = (input_shape[1]//8) * (input_shape[2]//8)

        self.layers_rnd = [
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
          
            nn.Flatten(),   

            nn.Linear(64*fc_size, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512)        
        ] 
        
         
        self.layers_rnd_target = [
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
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

        for i in range(len(self.layers_rnd)):
            if hasattr(self.layers_rnd[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_rnd[i].weight, 2.0**0.5)
                torch.nn.init.zeros_(self.layers_rnd[i].bias)

        for i in range(len(self.layers_rnd_target)):
            if hasattr(self.layers_rnd_target[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_rnd_target[i].weight, 2.0**0.5)
                torch.nn.init.zeros_(self.layers_rnd_target[i].bias)

        
        #coupled orthogonal init
        for i in range(len(self.layers_rnd_target)):
            if hasattr(self.layers_rnd_target[i], "weight"):
                wa, wb = self._coupled_ortohogonal_init(self.layers_rnd[i].weight.shape, 2.0**0.5)

                self.layers_rnd[i].weight        = torch.nn.Parameter(wa, requires_grad = True)
                self.layers_rnd_target[i].weight = torch.nn.Parameter(wb, requires_grad = True)

                torch.nn.init.zeros_(self.layers_rnd[i].bias)
                torch.nn.init.zeros_(self.layers_rnd_target[i].bias)
        
                
        self.model_rnd = nn.Sequential(*self.layers_rnd)
        self.model_rnd.to(self.device)

        self.model_rnd_target = nn.Sequential(*self.layers_rnd_target)
        self.model_rnd_target.to(self.device)

        for param in self.model_rnd_target.parameters():
            param.requires_grad = False
        self.model_rnd_target.eval()

        print("model_rnd")
        print(self.model_rnd)
        print(self.model_rnd_target)
        print("\n\n")

    def forward(self, state): 
        x = state[:,0:3,:,:]
        return self.model_rnd(x), self.model_rnd_target(x).detach()

    def save(self, path):
        torch.save(self.model_rnd.state_dict(), path + "model_rnd.pt")
        torch.save(self.model_rnd_target.state_dict(), path + "model_rnd_target.pt")
        

    def load(self, path):
        self.model_rnd.load_state_dict(torch.load(path + "model_rnd.pt", map_location = self.device))
        self.model_rnd_target.load_state_dict(torch.load(path + "model_rnd_target.pt", map_location = self.device))

        self.model_rnd.eval() 
        self.model_rnd_target.eval() 

    def _coupled_ortohogonal_init(self, shape, gain):
        w = torch.zeros((2*shape[0], ) + shape[1:])
        torch.nn.init.orthogonal_(w, gain)

        w = w.reshape((2, ) + shape)
        return w[0], w[1]

   
