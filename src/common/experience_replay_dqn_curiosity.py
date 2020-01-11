import numpy
import collections
import torch


Transition = collections.namedtuple("Transition", ("observation", "q_values", "action", "reward", "done"))


class Buffer():

    def __init__(self, size, gamma, observation_shape, actions_count):
        self.size   = size
        self.gamma  = gamma
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

    def create_indices(self, batch_size):
        self.indices = numpy.zeros(batch_size, dtype=int)
        for i in range(0, batch_size):
            self.indices[i] = numpy.random.randint(self.length() - 1)

    def get_icm_input(self, device):
        batch_size      = len(self.indices)

        observation_shape = self.buffer[0].observation.shape
        state_shape   = (batch_size, ) + observation_shape[0:]
        actions_count = len(self.buffer[0].q_values)

        q_values_shape = (batch_size, ) + (actions_count, )

        
        input           = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        input_next      = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        actions_one_hot = torch.zeros(q_values_shape,  dtype=torch.float32).to(device)


        for i in range(0, batch_size):
            n               = self.indices[i]
    
            input[i]        = torch.from_numpy(self.buffer[n].observation).to(device)
            input_next[i]   = torch.from_numpy(self.buffer[n+1].observation).to(device)
            
            action = self.buffer[n].action
            actions_one_hot[i][action]  = 1.0 


        return input, input_next, actions_one_hot

    def get_q_target(self, curiosity, alpha, device):
        batch_size      = len(self.indices)

        observation_shape = self.buffer[0].observation.shape
        actions_count = len(self.buffer[0].q_values)

        q_values_shape = (batch_size, ) + (actions_count, )

        target      = torch.zeros(q_values_shape,  dtype=torch.float32).to(device)

        for i in range(0, batch_size):
            n               = self.indices[i]

            if self.buffer[n].done:
                gamma_ = 0.0
            else: 
                gamma_ = self.gamma

            q_values    = self.buffer[n].q_values.copy()
            action      = self.buffer[n].action

            if self.buffer[n].done:
                reward = (1.0 - alpha)*self.buffer[n].reward
            else:
                reward = (1.0 - alpha)*self.buffer[n].reward + alpha*curiosity[i]

            q_values[action] = reward + gamma_*numpy.max(self.buffer[n+1].q_values)

            target[i] = torch.from_numpy(q_values).to(device)

        return target

