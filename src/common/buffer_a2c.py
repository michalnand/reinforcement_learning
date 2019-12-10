import numpy
import torch


class Buffer():

    def __init__(self, size):
        self.size   = size                
        self.clear()

    def clear(self):
        self.observations_v = []
        self.actions_v      = []
        self.rewards_v      = []
        self.dones_v        = []


    def length(self):
        return len(self.observations_v)

    def is_full(self):
        if self.length() >= self.size:
            return True
        else:
            return False

    def add(self, observation, action, reward, done, device):
        if self.is_full() == False:
            self.observations_v.append(observation.copy())
            self.actions_v.append(action)
            self.rewards_v.append(reward)
            self.dones_v.append(done)

    def _print(self):
        for i in range(self.length()):
            #print(self.observations_v[i], end = " ")
            print(self.actions_v[i], end = " ")
            print(self.rewards_v[i], end = " ")
            print(self.dones_v[i], end = " ")
            print("\n")

    def get(self, device):
        return torch.FloatTensor(self.observations_v).to(device),  torch.FloatTensor(self.actions_v).to(device), self.rewards_v, self.dones_v
