import torch
import torch.nn as nn

class MyModel(torch.nn.Module):
    def __init__(self, inputs_count, outputs_count):
        super(MyModel, self).__init__()

        self.l0     = nn.Linear(inputs_count, 1024)
        self.l1     = nn.ReLU()

        self.l2     = nn.Linear(1024, 256)
        self.l3     = nn.ReLU()

        self.l4     = nn.Linear(256, outputs_count)

    def forward(self, input):
        x = input

        x = self.l0.forward(x)
        x = self.l1.forward(x)
        x = self.l2.forward(x)
        x = self.l3.forward(x)
        x = self.l4.forward(x)

        return x


class CrazyModel(torch.nn.Module):
    def __init__(self, inputs_count, outputs_count):
        super(CrazyModel, self).__init__()

        self.l0     = nn.Linear(inputs_count, 1024)
        self.l1     = nn.ReLU()

        self.l0B     = nn.Linear(inputs_count, 256)
        self.l1B     = nn.Sigmoid()

        self.l2     = nn.Linear(1024, 256)
        self.l3     = nn.ReLU()

        self.l4     = nn.Linear(256, outputs_count)

    def forward(self, input):
        x = self.l0.forward(input)
        x = self.l1.forward(x)

        xb = self.l0B.forward(input)
        xb = self.l1B.forward(xb)

        x = self.l2.forward(x)
        x = self.l3.forward(x)
        x = self.l4.forward(x*(xb - 0.5))

        return x


inputs_count  = 768
outputs_count = 10
batch_size    = 32

#create model
model = CrazyModel(inputs_count, outputs_count)

#create solver, use ADAM, learning rate = 0.001
optimizer      = torch.optim.Adam(model.parameters(), lr= 0.001)

#some random data, just example
x_input     = torch.rand((batch_size, inputs_count))
y_target    = torch.rand((batch_size, outputs_count))

#network output
y_predicted = model.forward(x_input)



#clear gradients
optimizer.zero_grad()

#compute loss, RMS
loss   = ((y_target - y_predicted)**2).mean()

#backpropagation
loss.backward()

#update weights
optimizer.step()

print("loss = ", loss)


import numpy
import time
import gym

env = gym.make("LunarLander-v2")
env.reset()

actions_count   = env.action_space.n


while True:
    action = numpy.random.randint(actions_count)
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)
