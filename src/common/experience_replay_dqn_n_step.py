import numpy
import collections
import torch


Transition = collections.namedtuple("Transition", ("observation", "q_values", "action", "reward", "done"))


class Buffer():

    def __init__(self, size, gamma, steps, observation_shape, actions_count):
        self.size   = size
        self.gamma  = gamma
        self.steps  = steps
        self.observation_shape = observation_shape
        self.actions_count = actions_count
        
        self.ptr = 0
        self.buffer = []

    def _init_zeros(self):
        for _ in range(0, self.size):
            observation = numpy.zeros(self.observation_shape)
            q_values    = numpy.zeros(self.actions_count)
            self.buffer.append(Transition(observation, q_values, 0, 0.0, True))


    def length(self):
        return len(self.buffer)


    def add(self, observation, q_values, action, reward, done):
        if self.length() == 0:
            self._init_zeros()

        self.buffer[self.ptr] = Transition(observation.copy(), q_values.copy(), action, reward, done)
        self.ptr = (self.ptr+1)%self.size

    def _print(self):
        for i in range(self.length()):
            #print(self.buffer[i].observation, end = " ")
            print(self.buffer[i].q_values, end = " ")
            print(self.buffer[i].action, end = " ")
            print(self.buffer[i].reward, end = " ")
            print(self.buffer[i].done, end = " ")
            print("\n")

   
    def get_random_batch(self, batch_size, device):
        
        observation_shape = self.buffer[0].observation.shape
        state_shape   = (batch_size, ) + observation_shape[0:]
        actions_count = len(self.buffer[0].q_values)

        q_values_shape = (batch_size, ) + (actions_count, )

        input       = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        target      = torch.zeros(q_values_shape,  dtype=torch.float32).to(device)
 
        for i in range(0, batch_size):
            n      = numpy.random.randint(self.length() - self.steps - 1)

            gamma_ = self.gamma

            reward_sum = 0.0
            for k in range(self.steps):
                if self.buffer[n + k].done:
                    gamma_ = 0.0
                reward_sum+= self.buffer[n + k].reward*(gamma_**k)
            
            if self.buffer[n + self.steps].done:
                gamma_ = 0.0

            q_values    = self.buffer[n].q_values.copy()
            action      = self.buffer[n].action

            q_values[action] = reward_sum + gamma_*numpy.max(self.buffer[n + self.steps].q_values)
            
            input[i]  = torch.from_numpy(self.buffer[n].observation).to(device)
            target[i] = torch.from_numpy(q_values).to(device)
            
        return input, target

        