import gym
import numpy

class Create:
    def __init__(self, size_ = 9):
        self.size = size_
        self.env = gym.make('gym_go:go-v0', size=self.size)
        self.reset()


    def reset(self):
        self.observation = self.env.reset()
        self.legal_moves = self.find_legal_moves()
        return self.observation

        
    def step(self, action):
        self.observation, reward, done, info = self.env.step(action)
        self.legal_moves = self.find_legal_moves()

        return self.observation, reward, done, info

    def render(self):
        self.env.render()

    def get_active_player(self):
        print(self.observation[2])
        if self.observation[2][0][0] == 0:
            return 1
        else:
            return -1


    def e_greedy_move(self, q_values, epsilon):
        q_shifted = q_values - numpy.min(q_values) + (10.0**-7)

        #mask q_values with legal moves
        masked_moves = self.legal_moves*q_shifted

        move = -1
        if numpy.random.rand() < epsilon:
            #choose random move
            non_zero    = numpy.nonzero(masked_moves)[0]
            move        = numpy.random.choice(non_zero)
        else:
            #choose best move
            move = numpy.argmax(masked_moves)
        
        return int(move)



    def find_legal_moves(self):
        legal_moves = numpy.zeros(self.size**2 + 1)

        for y in range(self.size):
            for x in range(self.size):
                if self.observation[3][y][x] == 0:
                    legal_moves[self.pos_to_move_idx(y, x)] = 1
                    
        legal_moves[self.pass_move_idx()] = 1

        return legal_moves

    def pos_to_move_idx(self, y, x):
        return y*self.size + x

    def pass_move_idx(self):
        return self.size**2