import gym
import common.state
import numpy

class Create():

    def __init__(self, name = "Pong-V0", state_shape = (96, 96), frame_stacking = 4):

        print("creating env : ", name)

        self.observation    = None
        self.reward         = 0
        self.done           = False
        self.info           = None
        self.actions_count  = 0
        self.shape          = (0, 0, 0)

        self.env = gym.make(name) 
        self.reset()

        original_state_shape = self.env.observation_space.shape
        self.actions_count   = self.env.action_space.n

        self.state_converter    = common.state.State(original_state_shape, state_shape, frame_stacking)

        self.observation = self.state_converter.get()
        self.shape       = self.state_converter.get_shape()

        print("actions_count ", self.actions_count)
        print("env ready")

    def step(self, action):
        observation, self.reward, self.done, self.info = self.env.step(action)

        self.state_converter.update_state(observation)
        self.observation = self.state_converter.get()

        self.reward = numpy.clip(-10.0, 10.0, self.reward)

        return (self.observation, self.reward, self.done, self.info)

    def reset(self):
        self.env.reset()
        _, _, done, _ = self.env.step(1)
        
        if done:
            self.env.reset()
        
        _, _, done, _ = self.env.step(2)
        
        if done:
            self.env.reset()

    def render(self):
        self.env.render()

    def show_state(self):
        self.state.show()
