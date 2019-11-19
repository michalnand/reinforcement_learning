import numpy
import torch

import agents.agent_stats

import common.experience_replay
import common.experience_replay_dqn


import common.atari_wrapper

import sys
numpy.set_printoptions(threshold=sys.maxsize)


def loss_mse(y_target, y_hat):
    return torch.mean( (y_target - y_hat).pow(2) )

 
class Agent():
    def __init__(self, env, model, config, save_path = None, save_stats = True):
        self.env = env
        self.save_path = save_path

        self.action = 0

        self.batch_size     = config.batch_size

        self.exploration    = config.exploration
        self.gamma          = config.gamma

        self.experience_replay = common.experience_replay.Buffer(config.experience_replay_size)

        self.observation_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        self.model      = model.Model(self.observation_shape, self.actions_count)
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.observation    = env.reset()
        self.enable_training()

        self.iterations = 0

        self.score = 0

        if save_path != None and save_stats:
            self.training_stats = agents.agent_stats.AgentStats(self.save_path + "result/training")
            self.testing_stats  = agents.agent_stats.AgentStats(self.save_path + "result/testing")

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
        
        q_values = self.model.get_q_values(self.observation)
        self.action = self.choose_action_e_greedy(q_values, epsilon)

        observation_new, self.reward, self.done, self.info = self.env.step(self.action)

        if self.enabled_training:
            if self.experience_replay.is_full() == False:
                self.experience_replay.add(self.observation, q_values, self.action, self.reward, self.done)
            else:   
                self.train_model()

        self.observation = observation_new

        if hasattr(self, "training_stats") and hasattr(self, "testing_stats"):
            if self.enabled_training:
                self.training_stats.add(self.reward, self.done)
            else:
                self.testing_stats.add(self.reward, self.done)
            
            
        if self.done:
            self.env.reset()

        self.iterations+= 1
        self.score+= self.reward
        
        
    def train_model(self):
        self.experience_replay.compute(self.gamma)
                
        batches_count = self.experience_replay.length()//self.batch_size

        for _ in range(0, batches_count):
            input, target = self.experience_replay.get_random_batch(self.batch_size, self.model.device)
            
            output = self.model.forward(input)

            loss   = loss_mse(target, output)
    
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-10.0, 10.0)
            self.optimizer.step()

        self.experience_replay.clear()              

        
    def choose_action_e_greedy(self, q_values, epsilon):
        result = numpy.argmax(q_values)
        
        if numpy.random.random() < epsilon:
            result = numpy.random.randint(len(q_values))
        
        return result

    def save(self):
        self.model.save(self.save_path)

    def load(self):
        self.model.load(self.save_path)