import go_wrapper
import numpy



PLAYER_TRAINING     = 0
PLAYER_REFERENCE    = 1
PLAYER_BLACK        = 0
PLAYER_WHITE        = 1

class Create:

    def __init__(self, model, board_size = 9):
        self.env = go_wrapper.Create(size)

        self.result_stats           = numpy.zeros((2, 2))
        self.result_stats_smoothed  = numpy.zeros((2, 2))


    def play(self, games_count, batch_size = 100):
        
        for game in range(games_count//2):
            result_a = self.play_game(self.player_training, self.player_reference)
            result_b = self.play_game(self.player_reference, self.player_training)

            self.add_stats(result_a, result_b)

          
    def play_game(self, player_black, player_white):
        observation = self.env.reset()

        black_done = False
        white_done = False

        while black_done == False and white_done == False:
            q_values, epsilon = player_black.get_policy(observation)
            action = self.env.e_greedy_move(q_values, epsilon)

            observation, reward, black_done, info = self.env.step(action)
            player_black.add(observation, action, reward, black_done, info, q_values)

 
            q_values, epsilon = player_white.get_policy(observation)
            action = self.env.e_greedy_move(q_values, epsilon)

            observation, reward, white_done, info = self.env.step(action)
            player_black.add(observation, action, -1.0*reward, white_done, info, q_values)

        if reward > 0:
            result = 1     #black wins
        else:
            result = 0     #white wins
 
        return result

    def add_stats(self, result_a, result_b):
        k = 0.99

        if result_a > 0:
            self.result_stats[PLAYER_TRAINING][PLAYER_BLACK]+= 1
            self.result_stats_smoothed[PLAYER_TRAINING][PLAYER_BLACK] = k*self.result_stats_smoothed[PLAYER_TRAINING][PLAYER_BLACK] + (1.0 - k)*1.0
        else: 
            self.result_stats[PLAYER_REFERENCE][PLAYER_WHITE]+= 1
            self.result_stats_smoothed[PLAYER_REFERENCE][PLAYER_WHITE] = k*self.result_stats_smoothed[PLAYER_REFERENCE][PLAYER_WHITE] + (1.0 - k)*1.0

        if result_b > 0:
            self.result_stats[PLAYER_REFERENCE][PLAYER_BLACK]+= 1
            self.result_stats_smoothed[PLAYER_REFERENCE][PLAYER_BLACK] = k*self.result_stats_smoothed[PLAYER_REFERENCE][PLAYER_BLACK] + (1.0 - k)*1.0
        else: 
            self.result_stats[PLAYER_TRAINING][PLAYER_WHITE]+= 1
            self.result_stats_smoothed[PLAYER_TRAINING][PLAYER_WHITE] = k*self.result_stats_smoothed[PLAYER_TRAINING][PLAYER_WHITE] + (1.0 - k)*1.0
