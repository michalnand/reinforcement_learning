import numpy
import collections
import torch


Transition = collections.namedtuple("Transition", ("observation", "q_values", "action", "reward", "done"))


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

    def add(self, observation, q_values, action, reward, done):
        self.buffer.append(Transition(observation.copy(), q_values.copy(), action, reward, done))

    def _print(self):
        for i in range(self.length()):
            #print(self.buffer[i].observation, end = " ")
            print(self.buffer[i].q_values, end = " ")
            print(self.buffer[i].action, end = " ")
            print(self.buffer[i].reward, end = " ")
            print(self.buffer[i].done, end = " ")
            print("\n")

   
    def get_random_batch(self, gamma, batch_size, device):
        
        observation_shape = self.buffer[0].observation.shape

        if len(observation_shape) == 1:
            state_shape   = (batch_size, ) + observation_shape[0:]
        else:
            state_shape   = (batch_size, ) + observation_shape[1:]
        actions_count = len(self.buffer[0].q_values)

        q_values_shape = (batch_size, ) + (actions_count, )


        input   = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        target  = torch.zeros(q_values_shape,  dtype=torch.float32).to(device)
 
        for i in range(0, batch_size):
            n      = numpy.random.randint(self.length()-1)

            if self.buffer[n].done:
                gamma_ = 0.0
            else: 
                gamma_ = gamma
    
            q_values    = self.buffer[n].q_values.copy()
            action      = self.buffer[n].action

            q_values[action] = self.buffer[n].reward + gamma_*numpy.max(self.buffer[n+1].q_values)
            
            input[i]  = torch.from_numpy(self.buffer[n].observation).to(device)
            target[i] = torch.from_numpy(q_values).to(device)

        return input, target

        