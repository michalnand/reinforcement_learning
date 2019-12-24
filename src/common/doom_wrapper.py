from vizdoom import *
import collections
import numpy
import time
import cv2


ActionSpace      = collections.namedtuple("ActionSpace", "n")
ObservationSpace = collections.namedtuple("ObservationSpace", "shape")

class InitEnv():
    def __init__(self, mode = "basic", width = 96, height = 96, frame_stacking = 4):
        self.width  = width
        self.height = height
        self.frame_stacking = frame_stacking

        self.shape           = (self.frame_stacking, self.height, self.width)

        self.slices = numpy.zeros(self.shape)

        self.game = DoomGame()

        self.mode = mode

        self.reward_scale = 1.0

        if self.mode == "basic":
            self.reward_scale = 0.01
            self.game.load_config("common/doom_scenarios/basic.cfg")

            left    = [1, 0, 0]
            right   = [0, 1, 0]
            attack  = [0, 0, 1]

            self.actions = [left, right, attack]


        if self.mode == "defend_the_line":
            self.reward_scale = 1.0

            self.game.load_config("common/doom_scenarios/defend_the_line.cfg")

            left    = [1, 0, 0]
            right   = [0, 1, 0]
            attack  = [0, 0, 1]

            self.actions = [left, right, attack]
            
        self.action_space       = ActionSpace(len(self.actions))
        self.observation_space  = ObservationSpace(shape = self.shape)
        self.game.init()
        self.game.new_episode()

    def step(self, action):
        reward = self.game.make_action(self.actions[action])
        state = self.game.get_state()

        if self.game.is_episode_finished():
            done = True
            self.game.new_episode()
            state = self.game.get_state()
        else:
            done = False

        reward = reward*self.reward_scale

        if reward > 1.0:
            reward = 1.0
        if reward < -1.0:
            reward = -1.0
        
        return self._compute_state(state.screen_buffer), reward, [done, done], None

    def reset(self):
        self.game.new_episode()
        return self._compute_state(self.game.get_state().screen_buffer)

    def _compute_state(self, state):

        state = numpy.rollaxis(state, 2)
        state = numpy.rollaxis(state, 2)

        frame = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        for i in reversed(range(self.frame_stacking-1)):
            self.slices[i+1] = self.slices[i].copy()
        
        self.slices[0] = numpy.array(frame).copy()/255.0

        return self.slices

    def render(self):
        pass


def Create(mode = "basic", width = 96, height = 96, frame_stacking = 4):
    env = InitEnv(mode, width, height, frame_stacking)
    return env