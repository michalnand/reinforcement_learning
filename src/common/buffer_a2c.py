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

        value_shape         = (batch_size, ) + (1, )

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


    '''
    def compute(self):
        for n in reversed(range(self.length() - 1)):
            if self.buffer[n].done == True:
                gamma_ = 0.0
            else:
                gamma_ = self.gamma

            self.buffer[n].target_value[0] = self.buffer[n].reward + gamma_*self.buffer[n+1].target_value[0]
    '''
    
    '''
    def get_random_batch(self, batch_size, device):
        
        observation_shape = self.buffer[0].observation.shape
        state_shape   = (batch_size, ) + observation_shape[0:]

        value_shape = (batch_size, ) + (1, )

        input  = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        value  = torch.zeros(value_shape,  dtype=torch.float32).to(device)

        action = []
 
        for i in range(0, batch_size):
            n      = numpy.random.randint(self.length())
           
            input[i]  = torch.from_numpy(self.buffer[n].observation).to(device)
            value[i]  = torch.from_numpy(self.buffer[n].value).to(device)
            action.append(self.buffer[n].action)

        return input, value, action
    '''
