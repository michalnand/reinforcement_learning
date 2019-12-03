import numpy
import collections
import torch


Transition = collections.namedtuple("Transition", ("observation", "action", "reward",  "done"))

class Buffer():

    def __init__(self, size):
        self.size   = size                
        self.clear()

    def clear(self):
        self.buffer = []

    def length(self):
        return len(self.buffer)

    def is_full(self):
        if self.length() >= self.size:
            return True
        else:
            return False

    def add(self, observation, action, reward, done):
        if self.is_full() == False:
            self.buffer.append(Transition(observation.copy(), action, reward, done))

    def _print(self):
        for i in range(self.length()):
            #print(self.buffer[i].observation, end = " ")
            print(self.buffer[i].action, end = " ")
            print(self.buffer[i].reward, end = " ")
            print(self.buffer[i].done, end = " ")
            print("\n")

    def get(self, device):
        batch_size          = len(self.buffer)

        observation_shape   = self.buffer[0].observation.shape
        state_shape         = (batch_size, ) + observation_shape[0:]

        observation  = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        action       = numpy.zeros(batch_size, dtype=int)
        reward       = numpy.zeros(batch_size)
        done         = numpy.zeros(batch_size, dtype=bool)

        for n in range(batch_size):
            observation[n]  = torch.from_numpy(self.buffer[n].observation).to(device)
            action[n]       = self.buffer[n].action
            reward[n]       = self.buffer[n].reward
            done[n]         = self.buffer[n].done

        return observation, action, reward, done
