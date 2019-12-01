import time

class AgentStats():
    def __init__(self, log_file_name_prefix):
        self.log_file_name_prefix = log_file_name_prefix
        self.clear()


    def clear(self):
        self.iterations  = 0
        self.total_score = 0
        
        self.game_iterations_ = 0
        self.game_iterations = 0
        self.game_iterations_smooth = None

        self.game_score_ = 0
        self.game_score  = 0
        self.game_score_smooth = None
        
        self.game_id     = 0

        self.time_now    = time.time()
        self.time_prev   = self.time_now - 1.0
        self.fps_smooth  = 0.0

        f = open(self.log_file_name_prefix + "_progress.log", "w")
        f.close()

        f = open(self.log_file_name_prefix + "_experiment.log", "w")
        f.close()

    def add(self, reward, done):

        self.iterations+= 1
        self.total_score+= reward

        self.game_iterations_+= 1
        self.game_score_+= reward

        
        self.time_prev   = self.time_now
        self.time_now    = time.time()
        k = 0.01
        self.fps_smooth = (1.0 - k)*self.fps_smooth + k*1.0/(self.time_now - self.time_prev)


        if done:
            k = 0.01
            self.game_id+= 1

            self.game_iterations  = self.game_iterations_
            self.game_iterations_ = 0

            if self.game_iterations_smooth == None:
                self.game_iterations_smooth = self.game_iterations

            self.game_iterations_smooth = (1.0 - k)*self.game_iterations_smooth + k*self.game_iterations

            self.game_score  = self.game_score_
            self.game_score_ = 0

            if self.game_score_smooth == None:
                self.game_score_smooth = self.game_score

            self.game_score_smooth = (1.0 - k)*self.game_score_smooth + k*self.game_score

           
        if self.iterations%2000 == 0:
            s = str(self.iterations) + " "
            s+= str(self.game_id)   + " " 
            s+= str(self.total_score) + " " 
            s+= str(self.game_iterations) + " "
            s+= str(self.game_iterations_smooth) + " "
            s+= str(self.game_score) + " "
            s+= str(self.game_score_smooth) + " "
            s+= str(self.fps_smooth) + " "
            s+= "\n"
            
            f = open(self.log_file_name_prefix + "_progress.log", "a+")
            f.write(s)

        if done:
            s = "iterations                = " + str(self.iterations) + "\n"
            s+= "game_id                   = " + str(self.game_id) + "\n"
            s+= "score                     = "   + str(self.total_score) + "\n"
            s+= "game_iterations           = " + str(self.game_iterations) + "\n"
            s+= "game_iterations_smooth    = " + str(self.game_iterations_smooth) + "\n"
            s+= "game_score                = " + str(self.game_score) + "\n"
            s+= "game_score_smooth         = " + str(self.game_score_smooth) + "\n"
            s+= "fps_smooth                = " + str(self.fps_smooth) + "\n"

            s+= "\n\n\n"

            f = open(self.log_file_name_prefix + "_experiment.log", "a+")
            f.write(s)

            print(s)
    