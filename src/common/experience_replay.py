import numpy
import collections
import torch


Transition = collections.namedtuple("Transition", ("observation", "q_values", "action", "reward", "done"))



class Buffer():

    def __init__(self, size, gamma):
        self.size   = size
        self.gamma  = gamma
        self.clear()

    def clear(self):
        self.buffer = []

    def length(self):
        return len(self.buffer)

    def is_full(self):
        if self.length() >= self.size:
            return True
        return False

    def add(self, observation, q_values, action, reward, done):
        self.buffer.append(Transition(observation, q_values, action, reward, done))

    def _print(self):
        for i in range(self.length()):
            print(self.buffer[i].observation, end = " ")
            print(self.buffer[i].q_values, end = " ")
            print(self.buffer[i].action, end = " ")
            print(self.buffer[i].reward, end = " ")
            print(self.buffer[i].done, end = " ")
            print("\n")

    def compute(self):
        for n in reversed(range(self.length() - 1)):
            if self.buffer[n].done:
                gamma_ = 0.0
            else:
                gamma_ = self.gamma

            q_next = numpy.max(self.buffer[n+1].q_values)
            q_new  = self.buffer[n].reward + gamma_*q_next

            action = self.buffer[n].action

            self.buffer[n].q_values[action] = q_new

    def get_random_batch(self, batch_size, device):
        
        observation_shape = self.buffer[0].observation.shape
        state_shape   = (batch_size, ) + observation_shape[0:]
        actions_count = len(self.buffer[0].q_values)

        q_values_shape = (batch_size, ) + (actions_count, )


        input   = torch.zeros(state_shape,  dtype=torch.float32, requires_grad=False).to(device)
        target  = torch.zeros(q_values_shape,  dtype=torch.float32, requires_grad=False).to(device)
 
        for i in range(0, batch_size):
            n      = numpy.random.randint(self.length() -)
            
            input[i]  = torch.from_numpy(self.buffer[n].observation).to(device)
            target[i] = torch.from_numpy(self.buffer[n].q_values).to(device)

            input[i]  = input[i].float()
            target[i] = target[i].float()

        return input, target

        