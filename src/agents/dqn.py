import numpy
import torch

import common.experience_replay



def loss_mse(y_target, y_hat):
    return torch.mean( (y_target - y_hat).pow(2) )
    #return torch.sqrt(torch.mean((y_target - y_hat).pow(2)))
 
class Agent():
    def __init__(self, env, model, config, save_path):
        self.env = env
        self.save_path = save_path

        self.action = 0

        self.batch_size     = config.batch_size

        self.epsilon        = config.epsilon
        self.epsilon_end    = config.epsilon_end
        self.epsilon_decay  = config.epsilon_decay
        self.gamma          = config.gamma

        self.experience_replay = common.experience_replay.Buffer(config.experience_replay_size)

        self.iterations     = 0
        self.games_played   = 0
        self.score          = 0.0

        self.observation_shape = self.env.observation_space.shape
        self.actions_count     = self.env.env.action_space.n

        self.model      = model.Model(self.observation_shape, self.actions_count)
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)
        self.loss       = torch.nn.MSELoss()

        self.observation    = env.reset()
        self.actions_stats  = numpy.zeros(self.actions_count)

        self.enable_training()

        f = open(self.save_path + "result/training.log", "w")
        f.close()



    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self):
        if self.enabled_training:
            if self.epsilon > self.epsilon_end:
                self.epsilon*= self.epsilon_decay
                
            epsilon = self.epsilon
        else:
            epsilon = self.epsilon_end

        q_values = self.model.get_q_values(self.observation)
        self.action = self.choose_action_e_greedy(q_values, epsilon)

        observation_new, self.reward, self.done, self.info = self.env.step(self.action)

        if self.enabled_training:
            if self.experience_replay.is_full() == False:
                self.experience_replay.add(self.observation, q_values, self.action, self.reward, self.done)
            else:   
                self.train_model()
           

        self.observation = observation_new
            
        self.actions_stats[self.action]+= 1
        self.iterations+= 1
        self.score+= self.reward

        if self.done:
            self.env.reset()
            self.games_played+= 1

        
    def train_model(self):
        self.experience_replay.compute(self.gamma)
                
        batches_count = self.experience_replay.length()//self.batch_size

        loss_sum = 0
        for i in range(0, batches_count):
            input, target = self.experience_replay.get_random_batch(self.batch_size, self.model.device)
            
            output = self.model.forward(input)

            loss   = loss_mse(target, output)
            loss_sum+= torch.sum(loss).detach().to("cpu")
    
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-10.0, 10.0)
            self.optimizer.step()

        self.experience_replay.clear()                

        print(self.iterations, self.games_played, self.score, self.epsilon, loss_sum/batches_count)
        



    def softmax(self, values):
        m = numpy.max(values)

        result = numpy.exp(values - m)
        result/= numpy.sum(result)

        return result

    def choose_action(self, q_values):
        probs = self.softmax(q_values)
        actions = list(range(len(probs)))
        return numpy.random.choice(actions, p = probs)

    def choose_action_e_greedy(self, q_values, epsilon):
        result = numpy.argmax(q_values)
        

        if numpy.random.random() < epsilon:
            result = numpy.random.randint(len(q_values))

        
        return result

    def _print(self):
        f = open(self.save_path + "result/training.log", "a+")
        s = str(self.iterations) + " " + str(self.games_played) + " " + str(self.score) + " " + str(self.epsilon) + "\n"
        f.write(s)
        print(s)


    def save(self):
        self.model.save(self.save_path)

    def load(self):
        self.model.load(self.save_path)