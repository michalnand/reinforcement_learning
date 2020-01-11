import numpy
import torch

import agents.agent_stats

import common.experience_replay_dqn_curiosity


class Agent():
    def __init__(self, env, model, config, save_path = None, save_stats = True):
        self.env = env
        self.save_path = save_path

        self.action = 0

        self.batch_size     = config.batch_size

        self.exploration    = config.exploration
        self.alpha          = config.alpha
        self.beta1          = config.beta1
        self.beta2          = config.beta2
        self.beta3          = config.beta3
        self.gamma          = config.gamma

        self.update_frequency = config.update_frequency

       
        self.observation_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        self.experience_replay = common.experience_replay_dqn_curiosity.Buffer(config.experience_replay_size, self.gamma, self.observation_shape,  self.actions_count)

        self.model      = model.Model(self.observation_shape, self.actions_count)

        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.observation        = env.reset()
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

        observation_new, self.reward, done, self.info = self.env.step(self.action)

        round_done = done[0]
        game_done  = done[1]

        if self.enabled_training:
            self.experience_replay.add(self.observation, q_values, self.action, self.reward, round_done)

       
        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()

        self.observation = observation_new

        if hasattr(self, "training_stats") and hasattr(self, "testing_stats"):
            if self.enabled_training:
                self.training_stats.add(self.reward, game_done)
            else:
                self.testing_stats.add(self.reward, game_done)
            
            
        if game_done:
            self.env.reset()

        self.iterations+= 1
        self.score+= self.reward
        
        
    def train_model(self):
        self.optimizer.zero_grad()

        self.experience_replay.create_indices(self.batch_size)

        input, input_next, actions_target = self.experience_replay.get_icm_input(self.model.device)


        q_values, curiosity, actions_predicted = self.model.forward(input, input_next, actions_target)
        q_target = self.experience_replay.get_q_target(curiosity.detach().to("cpu").numpy(), self.alpha, self.model.device)


        loss_inverse    = ((actions_target - actions_predicted)**2).mean()
        loss_forward    = curiosity.mean()
        loss_q_values   = ((q_target - q_values)**2).mean()

        loss = self.beta1*loss_inverse + self.beta2*loss_forward + self.beta3*loss_q_values

        loss.backward()

        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()

        '''
        print("loss_inverse = ", loss_inverse.detach().to("cpu").numpy())
        print("loss_forward = ", loss_forward.detach().to("cpu").numpy())
        print("loss_q_values = ", loss_q_values.detach().to("cpu").numpy())
        print("loss = ", loss.detach().to("cpu").numpy())
        print("curiosity = ", curiosity.detach().to("cpu").numpy())
        print("\n\n\n")
        '''

        '''
        idx_actions_target      = torch.argmax(actions_target, dim = 1).detach()
        idx_actions_predicted   = torch.argmax(actions_predicted, dim = 1).detach()

        match = idx_actions_target == idx_actions_predicted

        print(match)
        print("\n\n\n")
        '''


        
    def choose_action_e_greedy(self, q_values, epsilon):
        result = numpy.argmax(q_values)
        
        if numpy.random.random() < epsilon:
            result = numpy.random.randint(len(q_values))
        
        return result

    def save(self):
        self.model.save(self.save_path)

    def load(self):
        self.model.load(self.save_path)