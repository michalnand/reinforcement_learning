import numpy
import collections

ActionSpace      = collections.namedtuple("ActionSpace", ("n"))
ObservationSpace = collections.namedtuple("ObservationSpace", ("shape"))


class EnvMulti():
    def __init__(self, envs, change_period = 4096):
        self.envs = envs
        self.change_period = change_period

        self.enable_auto_env_change()

        self._reset_all()


    def _reset_all(self):
        self.iterations  = 0
        self.current_env = 0

        for i in range(len(self.envs)):
            self.envs[i].reset()

        self.actions_count     = []
        for i in range(len(self.envs)):
            self.actions_count.append(self.envs[i].action_space.n)

        self.action_space      = ActionSpace(numpy.max(self.actions_count))
        self.observation_space = ObservationSpace(self.envs[0].observation_space.shape)

        return self.envs[self.current_env].reset()

    def reset(self):
        return self.envs[self.current_env].reset()

    def set_env(self, env_id):
        self.auto_env_change = False
        self.current_env = env_id

    def enable_auto_env_change(self):
        self.auto_env_change = True

    def disable_auto_env_change(self):
        self.auto_env_change = False

    def step(self, action):
        action = action%self.actions_count[self.current_env]

        observation, reward, done, info = self.envs[self.current_env].step(action)

        self.iterations+= 1
        if self.auto_env_change:
            if self.iterations%self.change_period == 0:
                self.current_env = numpy.random.randint(len(self.envs))
        
        return observation, reward, done, info

    def render(self):
        self.envs[self.current_env].render()