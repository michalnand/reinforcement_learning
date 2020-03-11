import numpy
import torch

import agents.agent_stats

import common.experience_replay_dqn_n_step_icm


class Agent():
    def __init__(self, env, model, config, save_path = None, save_stats = True):
        self.env = env
        self.save_path = save_path

        self.action = 0

        self.batch_size     = config.batch_size

        self.exploration    = config.exploration
        self.gamma          = config.gamma
        self.bellman_steps  = config.bellman_steps

        self.update_frequency = config.update_frequency

       
        self.observation_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        self.experience_replay = common.experience_replay_dqn_n_step_icm.Buffer(config.experience_replay_size, self.gamma, self.bellman_steps, self.observation_shape,  self.actions_count)

        self.model  = model.ModelDQNICM(self.observation_shape, self.actions_count)

        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr = config.learning_rate)

        self.observation    = env.reset()
        self.enable_training() 

        self.iterations = 0

        self.score = 0

        self.accuracy  = 0.0
        self.curiosity = 0.0

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
            print("\n\n")
            print("accuracy = ", self.accuracy)
            print("curiosity = ", self.curiosity)
            print("\n\n")

        self.iterations+= 1
        self.score+= self.reward        
        
    def train_model(self):

        state, state_next, q_target, actions_target = self.experience_replay.get_random_batch(self.batch_size, self.model.device)
        
        #train DQN model
        q_predicted, actions_predicted, curiosity = self.model.forward(state, state_next, actions_target)
        self.optimizer.zero_grad()

        #TODO -> add curiosity reward
        #q_target_     = q_target + curiosity.detach().unsqueeze(0)
 
        loss_q_values = ((q_target - q_predicted)**2).mean()
        loss_actions  = ((actions_target - actions_predicted)**2).mean()
        loss_curiosity= curiosity.mean()

        loss = 0.1*loss_q_values + 0.8*loss_actions + 0.01*loss_curiosity
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()

        actions_target_idx      = torch.argmax(actions_target, dim = 1).detach().to("cpu").numpy()
        actions_predicted_idx   = torch.argmax(actions_predicted, dim = 1).detach().to("cpu").numpy()

        hit = numpy.sum(actions_target_idx == actions_predicted_idx)
        miss= numpy.sum(actions_target_idx != actions_predicted_idx)        
        
        
        k = 0.98
        accuracy = 100.0*hit/(hit + miss)

        self.accuracy  = k*self.accuracy + (1.0 - k)*accuracy
        self.curiosity = k*self.curiosity + (1.0 - k)*curiosity[0]
        
  
    def choose_action_e_greedy(self, q_values, epsilon):
        result = numpy.argmax(q_values)
        
        if numpy.random.random() < epsilon:
            result = numpy.random.randint(len(q_values))
        
        return result

    def save(self):
        self.model.save(self.save_path)

    def load(self):
        self.model.load(self.save_path)