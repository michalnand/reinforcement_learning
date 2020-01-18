import numpy
import collections
import torch


Transition = collections.namedtuple("Transition", ("observation", "q_values", "q_target_values", "action", "reward", "done"))


class Buffer():

    def __init__(self, size, gamma, bellman_steps, observation_shape, actions_count):
        self.size   = size
        self.gamma  = gamma
        self.bellman_steps  = bellman_steps

        self.observation_shape = observation_shape
        self.actions_count = actions_count
        
        self.ptr = 0
        self.compute_ptr = 0
        self.buffer = []

    def _init_zeros(self):
        for _ in range(0, self.size):
            observation = numpy.zeros(self.observation_shape)
            q_values    = numpy.zeros(self.actions_count)
            q_target_values    = numpy.zeros(self.actions_count)
            self.buffer.append(Transition(observation, q_values, q_target_values, 0, 0.0, True))

        self.q_errors = numpy.zeros(self.size)



    def add(self, observation, q_values, action, reward, done):
        if self.size == 0:
            self._init_zeros()

        self.buffer[self.ptr] = Transition(observation.copy(), q_values.copy(), q_values.copy(), action, reward, done)
        self.ptr = (self.ptr+1)%self.size

    def _print(self):
        for i in range(self.size):
            #print(self.buffer[i].observation, end = " ")
            print(self.buffer[i].q_values, end = " ")
            print(self.buffer[i].q_target_values, end = " ")
            print(self.buffer[i].action, end = " ")
            print(self.buffer[i].reward, end = " ")
            print(self.buffer[i].done, end = " ")
            print("\n")

    def compute(self):
        while self.compute_ptr != self.ptr:    

            q_target = 0.0

            gamma_ = self.gamma
            for k in range(self.bellman_steps):
                idx = (self.compute_ptr + k)%self.size
                if self.buffer[idx].done:
                    gamma_ = 0.0
            
                q_target+= self.buffer[idx].reward*(gamma_**k)

            next_idx = (self.compute_ptr + self.bellman_steps)%self.size

            if self.buffer[next_idx].done:
                gamma_ = 0.0
            
            gamma_ = gamma_**self.bellman_steps
            
            action      = self.buffer[self.compute_ptr].action 

            target_q_value = q_target + gamma_*numpy.max(self.buffer[next_idx].q_values)
            
            self.buffer[self.compute_ptr].q_target_values[action] = target_q_value

            self.q_errors[self.compute_ptr] = numpy.mean((self.buffer[self.compute_ptr].q_target_values - self.buffer[self.compute_ptr].q_values)**2)

            self.compute_ptr = (self.compute_ptr + 1)%self.size

        self.probs = self.q_errors/numpy.sum(self.q_errors)
        
   
    def get_random_batch(self, batch_size, device):
        self.compute()
        
        observation_shape   = self.buffer[0].observation.shape
        state_shape         = (batch_size, ) + observation_shape[0:]
        actions_count       = len(self.buffer[0].q_values)

        q_values_shape = (batch_size, ) + (actions_count, )

        input       = torch.zeros(state_shape, dtype=torch.float32).to(device)
        target      = torch.zeros(q_values_shape, dtype=torch.float32).to(device)

        indices = numpy.random.choice(self.size, batch_size, p=self.probs)

        for i in range(0, batch_size):
            #todo - use self.probs for priority selection
            #n      = numpy.random.randint(self.size - 1 - self.bellman_steps)

            n = indices[i]

            input[i]  = torch.from_numpy(self.buffer[n].observation).to(device)
            target[i] = torch.from_numpy(self.buffer[n].q_target_values).to(device)
            
        return input, target

        