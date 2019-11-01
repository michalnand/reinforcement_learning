import numpy
import collections
import torch


Transition = collections.namedtuple("Transition", ("observation", "action", "reward", "state_value", "done"))



class Buffer():

    def __init__(self, size):
        self.size = size
        self.clear()

    def clear(self):
        self.buffer = []

    def length(self):
        return len(self.buffer)

    def is_full(self):
        if self.length() >= self.size:
            return True
        return False

    def add(self, observation, action, reward, done):
        self.buffer.append(Transition(observation, action, reward, numpy.zeros(1, dtype=float), done))

    def _print(self):
        for i in range(self.length()):
            print(self.buffer[i].observation, end = " ")
            print(self.buffer[i].action, end = " ")
            print(self.buffer[i].reward, end = " ")
            print(self.buffer[i].state_value, end = " ")
            print(self.buffer[i].done, end = " ")
            print("\n")

    def compute(self, gamma = 0.99):

        for n in reversed(range(self.length() - 1)):

            if self.buffer[n].done:
                gamma_ = 0.0
            else:
                gamma_ = gamma

            self.buffer[n].state_value[0] = gamma_*self.buffer[n+1].state_value[0] + self.buffer[n].reward


    def get_random_batch(self, batch_size, device):
        
        observation_shape = self.buffer[0].observation.shape

        if len(observation_shape) == 1:
            state_shape   = (batch_size, ) + observation_shape[0:]
        else:
            state_shape   = (batch_size, ) + observation_shape[1:]
        
        state_value_shape = (batch_size, 1)
        

        observation   = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        state_value   = torch.zeros(state_value_shape,  dtype=torch.float32).to(device)

        actions = numpy.zeros(batch_size, dtype=int)
 
        for i in range(0, batch_size):
            n      = numpy.random.randint(self.length())
            
            observation[i] = torch.from_numpy(self.buffer[n].observation).to(device)
            state_value[i] = torch.from_numpy(self.buffer[n].state_value).to(device)
            actions[i]     = self.buffer[n].action

        return observation, state_value, actions

        