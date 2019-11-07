import gym
import common.state
import numpy

class Create():

    def __init__(self, name = "Pong-v0", state_shape = (96, 96), frame_stacking = 4, frame_skiping = 2):

        print("creating env : ", name)

        self.observation_space    = None
        self.reward         = 0
        self.done           = False
        self.info           = None
        
        self.frame_skiping = frame_skiping
        
        self.env = gym.make(name) 

        original_state_shape = self.env.observation_space.shape
        self.action_space   = self.env.action_space

        self.state_converter    = common.state.State(original_state_shape, state_shape, frame_stacking)

        self.observation_space = self.state_converter.get()
        self.shape       = self.state_converter.get_shape()

        self.reset()

        print("actions_count ", self.action_space.n)
        print("env ready")

    def step(self, action):
        self.reward = 0.0
        for _ in range(self.frame_skiping):
            observation, reward, self.done, self.info = self.env.step(action)
            self.reward+= reward

        self.state_converter.update_state(observation)
        self.observation_space = self.state_converter.get()

        self.reward = numpy.clip(-10.0, 10.0, self.reward)

        return (self.observation_space, self.reward, self.done, self.info)

    def reset(self):
        self.env.reset()
        observation, _, done, _ = self.env.step(1)
        
        if done:
            self.env.reset()
        
        observation, _, done, _ = self.env.step(2)
        
        if done:
            self.env.reset()

        self.state_converter.update_state(observation)
        self.observation_space = self.state_converter.get()

        return self.observation_space

    def render(self):
        self.env.render()

    def show_state(self):
        self.state_converter.show()
