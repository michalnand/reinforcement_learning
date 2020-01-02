import sys
sys.path.append('..')

import numpy
import torch

import agents.agent_stats

import common.experience_replay_dqn


class Agent():
    def __init__(self, env, model, config, save_path = None, save_stats = True):
        self.env = env
        self.save_path = save_path

        self.action = 0

        self.batch_size     = config.batch_size

        self.exploration    = config.exploration
        self.gamma          = config.gamma

        self.update_frequency = config.update_frequency

       
        self.observation_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        self.experience_replay = common.experience_replay_dqn.Buffer(config.experience_replay_size, self.gamma, self.observation_shape,  self.actions_count)

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


    def get_policy(self, observation):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()

        return self.model.get_q_values(self.observation), epsilon

    def add(self, observation, action, reward, done, info, q_values):
        if self.enabled_training:
            self.experience_replay.add(observation, q_values, action, reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()

        if hasattr(self, "training_stats") and hasattr(self, "testing_stats"):
            if self.enabled_training:
                self.training_stats.add(reward, done)
            else:
                self.testing_stats.add(reward, done)

        self.iterations+= 1
        self.score+= reward

    def train_model(self):
        input, q_target = self.experience_replay.get_random_batch(self.batch_size, self.model.device)
            
        q_predicted = self.model.forward(input)

        self.optimizer.zero_grad()

        loss = ((q_target - q_predicted)**2).mean() 
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()