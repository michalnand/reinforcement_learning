import numpy
import collections


ActionSpace      = collections.namedtuple("ActionSpace", "n")
ObservationSpace = collections.namedtuple("ObservationSpace", "shape")


class EnvCart:

    def __init__(self, random_params = False):
        
        self.random_params  = random_params
        

        self.shape = (5,)

        

        self.actions            = [0.0, -1.0, 1.0, -0.1, 0.1]
        

        self.action_space       = ActionSpace(len(self.actions))
        self.observation_space  = ObservationSpace(shape = self.shape)

        self.reset()
    

    def reset(self):
        self.dt             = 0.01
        self.threshold      = 2.0*self.dt

        self.target_position = 0.9
        
        self.cart_position   = 0.0
        self.cart_velocity   = 0.0
      
        self.inertia         = 0.9
        self.cart_mass       = 1.0

        

        self.steps           = 0

        if self.random_params:
            self.target_position = numpy.random.rand()

            self.cart_velocity   = (2.0*numpy.random.rand() - 1.0)*0.1

            inertia_min          = 0.3
            inertia_max          = 0.9

            mass_min             = 0.1
            mass_max             = 1.0

            self.inertia         = (inertia_max - inertia_min)*numpy.random.rand()  + inertia_min
            self.cart_mass       = (mass_max - mass_min)*numpy.random.rand()        + mass_min

        self.done   = [False, False]
        self.reward = 0.0

        self.position0       = self.cart_position
        self.position1       = self.cart_position
        self.position2       = self.cart_position
        self.position3       = self.cart_position


        self._compute_observation()

        return self.observation

    def step(self, action):

        force = self.actions[action]
        
        acceleration = force/self.cart_mass

        self.cart_velocity = self.cart_velocity + self.inertia*(acceleration - self.cart_velocity)*self.dt      
        self.cart_position = self._saturate(self.cart_position + self.cart_velocity*self.dt, 0.0, 1.0)

        dist = numpy.abs(self.cart_position - self.target_position)
        velocity = numpy.abs(self.cart_velocity)

        self.steps+= 1

        self.reward = -0.0001

        self.done   = [False, False]

        if self.steps >= 1024:
            self.done = [True, True]
            self.reward = -1.0
            self.steps  = 0
        elif dist < self.threshold:
            self.reward = 1.0 - 0.9*velocity
            self.done   = [True, True]
        
        self._compute_observation()

        return self.observation, self.reward, self.done, "info"


    def _saturate(self, value, min, max):
        result = value
        if result > max:
            result = max
        if result < min:
            result = min
        
        return result

    def _compute_observation(self):
        '''
        self.observation = numpy.zeros(3)
        self.observation[0] = self.target_position
        self.observation[1] = self.cart_position
        self.observation[2] = self.cart_velocity
        '''
        self.observation = numpy.zeros(5)
        self.observation[0] = self.target_position
        self.observation[1] = self.position0
        self.observation[2] = self.position1
        self.observation[3] = self.position2
        self.observation[4] = self.position3

        self.position3 = self.position2
        self.position2 = self.position1
        self.position1 = self.position0
        self.position0 = self.cart_position
        

    def render(self):
        dist = numpy.abs(self.cart_position - self.target_position)
        print(self.observation, dist)

        points = 100

        cart_position   = (self.cart_position   + 1.0) /2.0
        target_position = (self.target_position + 1.0) /2.0

        cart_position   = round((points-1)*cart_position)
        target_position = round((points-1)*target_position)

        for i in range(points):
            if i == cart_position:
                print("C", end = "")
            elif i == target_position:
                print("T", end = "")
            else:
                print(".", end = "")
            
        print("\n\n")

'''    
env = EnvCart(True)
score = 0
while True:

    action = numpy.random.randint(3)

    observation, reward, done, _ = env.step(action)

    if done[0]:
        env.reset()

    env.render()

    score+= reward

    print(score)

'''