import numpy
import time
import random

import common.state

class Create():

    def __init__(self, size = 16):
        self.size           = size

        self.board          = None
        self.reward         = 0
        self.done           = False
        self.info           = None
        self.actions_count  = 0


        self.width          = size
        self.height         = size
        self.depth          = 3

        #self.shape          = (1, self.depth, self.height, self.width)

        self.state = common.state.State((self.width, self.height, self.depth), (self.width, self.height), 4)
        self.shape = self.state.get_shape()

        print("\n\nshape = ", self.shape, "\n\n")

        #init state, as 1D vector (tensor with size depth*height*width)
        self.board_init()

        #4 actions for movements
        self.actions_count  = 3
        self.player_size    = 3

        self.player_0_points = 0.0
        self.player_1_points = 0.0

        self.game_idx = 0

        self.reset()

    def board_init(self):
        self.board = numpy.zeros( (self.width, self.height, self.depth) )
        self.state.clear()


    def reset(self):
        #initial players position
        self.player_0 = self.height/2
        self.player_1 = self.height/2

        #ball to center + some noise
        self.ball_x  = self.width/2  + random.randint(0, 1)
        self.ball_y  = self.height/2 + random.randint(0, 1)

        #random ball move

        if (numpy.random.randint(1024)%2) == 0:
            self.ball_dx = 1
        else:
            self.ball_dx = -1

        if (numpy.random.randint(1024)%2) == 0:
            self.ball_dy = 1
        else:
            self.ball_dy = -1

        self.__position_to_state()

        return self.observation

  
    def render(self):

        #print(self.observation)

        for y in range(self.height):
            for x in range(self.width):
                v = self.observation[0][0][y][x]
                if v > 0:
                    print("* ", end = "")
                else:
                    print(". ", end = "")
            print("\n") 
        
        print("\n\n")

      

    def step(self, action):
        self.do_action(action)
        self.info = "pong"
        return (self.observation, self.reward, self.done, self.info)



    def do_action(self, action):
        self.reward = 0.0
        self.done = False

        if action == 0:
            player_0_dx = 1
        elif action == 1:
            player_0_dx = -1
        else:
            player_0_dx = 0

        self.player_0+= player_0_dx
        self.player_0 = numpy.clip(self.player_0, 0, self.height-1)

        if numpy.absolute(self.player_0 - self.ball_y) < self.player_size:
            player_0_hit = True
        else:
            player_0_hit = False



        if self.ball_y > self.player_1:
            player_1_dx = 1
        else:
            player_1_dx = -1

        if numpy.random.rand() < 0.12:
            player_1_dx*= -1

        self.player_1+= player_1_dx
        self.player_1 = numpy.clip(self.player_1, 0, self.height-1)

        if numpy.absolute(self.player_1 - self.ball_y) < self.player_size:
            player_1_hit = True
        else:
            player_1_hit = False



        if self.ball_x <= 1:
            if player_0_hit:
                self.ball_dx = 1
            else:
                self.reset()
                self.player_1_points+= 1
                self.reward = -1.0

        if self.ball_x >=  self.width-1:
            if player_1_hit:
                self.ball_dx = -1
            else:
                self.reset()
                self.player_0_points+= 1
                self.reward = +1.0

        if (self.player_0_points + self.player_1_points >= 64):
            self.player_0_points = 0
            self.player_1_points = 0
            self.done = True


        if self.ball_y <= 0:
            self.ball_dy = 1

        if self.ball_y >=  self.height-1:
            self.ball_dy = -1

        self.ball_x+= self.ball_dx
        self.ball_y+= self.ball_dy

        #print(self.get_score(), self.ball_y, self.player_0, self.player_1)

        self.__position_to_state()

    def __position_to_state(self):
        ball_x = self.__saturate(int(self.ball_x), 0, self.width-1)
        ball_y = self.__saturate(int(self.ball_y), 0, self.height-1)

        player_0 = self.__saturate(int(self.player_0), 0, self.height-1)
        player_1 = self.__saturate(int(self.player_1), 0, self.height-1)

        self.board.fill(0.0)

        self.board[ball_y][ball_x][0]                  = 255.0
        self.board[player_0][0][0]                     = 255.0
        self.board[player_1][self.width-1][0]          = 255.0

        self.state.update_state(self.board)
        self.observation = self.state.get()

    def __saturate(self, value, min, max):
        if value > max:
            value = max

        if value < min:
            value = min

        return value
