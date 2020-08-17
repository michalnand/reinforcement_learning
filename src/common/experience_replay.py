import numpy
import collections
import torch

Transition = collections.namedtuple("Transition", ("state", "action", "reward", "done"))

class Buffer():

    def __init__(self, size):
        self.size   = size
       
        self.ptr    = 0 
        self.buffer = []

    def length(self):
        return len(self.buffer)

    def is_full(self):
        if self.length() == self.size:
            return True
            
        return False

    def add(self, state, action, reward, done):

        if done != 0:
            done_ = 1.0
        else:
            done_ = 0.0

        item = Transition(state.copy(), action.copy(), reward, done_)

        if self.length() < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.ptr] = item
            self.ptr = (self.ptr + 1)%self.length()

    def _print(self):
        for i in range(self.length()):
            #print(self.buffer[i].state, end = " ")
            print(self.buffer[i].action, end = " ")
            print(self.buffer[i].reward, end = " ")
            print(self.buffer[i].done, end = " ")
            print("\n")

   
    def sample(self, batch_size, device):
        
        state_shape     = (batch_size, ) + self.buffer[0].state.shape[0:]
        action_shape    = (batch_size, ) + self.buffer[0].action.shape[0:]
        reward_shape    = (batch_size, )
        done_shape      = (batch_size, )


        state_t         = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        action_t        = torch.zeros(action_shape,  dtype=torch.float32).to(device)
        reward_t        = torch.zeros(reward_shape,  dtype=torch.float32).to(device)
        state_next_t    = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        done_t          = torch.zeros(done_shape,  dtype=torch.float32).to(device)

        
        for i in range(0, batch_size):
            n  = numpy.random.randint(self.length() - 1)
            state_t[i]      = torch.from_numpy(self.buffer[n].state).to(device)
            action_t[i]     = torch.from_numpy(self.buffer[n].action).to(device)
            reward_t[i]     = torch.from_numpy(numpy.asarray(self.buffer[n].reward)).to(device)
            state_next_t[i] = torch.from_numpy(self.buffer[n+1].state).to(device)
            done_t[i]       = torch.from_numpy(numpy.asarray(self.buffer[n].done)).to(device)

        return state_t, action_t, reward_t, state_next_t, done_t


if __name__ == "__main__":
    state_shape     = (3, 13, 17)
    action_shape    = (7,)


    replay_buffer = Buffer(107)

    for i in range(1000):
        state   = numpy.random.randn(state_shape[0], state_shape[1], state_shape[2])
        action  = numpy.random.randn(action_shape[0])
        reward  = numpy.random.rand(1)
        done    = numpy.random.randint(2)

        replay_buffer.add(state, action, reward, done)

        if i > 1:
            state_t, action_t, reward_t, state_next_t, done_t = replay_buffer.sample(5, device="cpu")


    print(state_t)
    print(action_t)
    print(reward_t)
    print(state_next_t)
    print(done_t)

