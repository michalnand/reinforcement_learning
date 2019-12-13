import numpy
import torch

import agents.agent_stats

from torch.distributions import Categorical

class Agent():
    def __init__(self, env, model, config, save_path = None, save_stats = True):
        self.env = env
        self.save_path = save_path
 
        self.gamma          = config.gamma
        self.entropy_beta   = config.entropy_beta
        self.batch_size     = config.batch_size
       
        self.observation_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n


        self.model          = model.Model(self.observation_shape, self.actions_count)

        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.observation    = env.reset()
        self.enable_training()

        self.iterations = 0

        self.score      = 0

        if save_path != None and save_stats:
            self.training_stats = agents.agent_stats.AgentStats(self.save_path + "result/training")
            self.testing_stats  = agents.agent_stats.AgentStats(self.save_path + "result/testing")

        self.init_buffer()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False


    def init_buffer(self):
        self.logits_b           = torch.zeros((self.batch_size, self.actions_count)).to(self.model.device)
        self.values_b           = torch.zeros((self.batch_size, 1)).to(self.model.device)
        self.action_b           = torch.LongTensor(self.batch_size)
        self.rewards_b          = numpy.zeros(self.batch_size)
        self.done_b             = numpy.zeros(self.batch_size, dtype=bool)

        self.idx = 0

    def main(self):              

        observation_t   = torch.tensor(self.observation, dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)
        logits, value   = self.model.forward(observation_t)

       
       
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
       
            
        self.observation, reward, done, _ = self.env.step(action_t.item())
        
        round_done = done[0]
        game_done  = done[1] 


        if self.enabled_training:
            self.logits_b[self.idx]     = logits.squeeze(0)
            self.values_b[self.idx]     = value.squeeze(0)
            self.action_b[self.idx]     = action_t.item()
            self.rewards_b[self.idx]    = reward
            self.done_b[self.idx]       = round_done
            self.idx+= 1

        if self.idx >= self.batch_size:
            

            
            #target_values_b = self._calc_q_valuesA(self.rewards_b, self.values_b.detach().cpu().numpy(), self.done_b)
            target_values_b = self._calc_q_valuesB(self.rewards_b, self.done_b)

            target_values_b = torch.FloatTensor(target_values_b).to(self.model.device)

            '''
            print("target_values_t ", target_values_b.shape)
            print("values_t ", self.values_b.shape)
            print("logits_t ", self.logits_b.shape)
            print("action_t ", self.action_b.shape)
            print("\n")
            '''


            probs     = torch.nn.functional.softmax(self.logits_b, dim = 1)
            log_probs = torch.nn.functional.log_softmax(self.logits_b, dim = 1)

            '''
            compute critic loss, as MSE : L = (T - V(s))^2
            '''
            loss_value = (target_values_b - self.values_b)**2
            loss_value = loss_value.mean()


            '''
            compute actor loss, L = log(pi(s, a))*(T - V(s)) = log(pi(s, a))*A
            TODO : log softmax is better for numerical stability
            '''
            advantage  = (target_values_b - self.values_b).detach()
            loss_policy = -log_probs[range(len(log_probs)), self.action_b]*advantage
            loss_policy = loss_policy.mean()

            '''
            compute entropy loss, to avoid greedy strategy
            L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
            '''
            loss_entropy = (probs*log_probs).sum(dim = 1)
            loss_entropy = self.entropy_beta*loss_entropy.mean()
            



            #train network, with gradient cliping
            #loss = loss_value + loss_policy + loss_entropy
            #loss.backward()

            loss_value.backward(retain_graph=True)
            loss_policy.backward(retain_graph=True)
            loss_entropy.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step() 
            self.optimizer.zero_grad()

            #clear batch buffer
            self.init_buffer()

            '''            
            print("loss_value = ", loss_value.detach().cpu().numpy())
            print("loss_policy = ", loss_policy.detach().cpu().numpy())
            print("loss_entropy = ", loss_entropy.detach().cpu().numpy())
            print("\n\n\n")
            '''

        if hasattr(self, "training_stats") and hasattr(self, "testing_stats"):
            if self.enabled_training:
                self.training_stats.add(reward, game_done)
            else:
                self.testing_stats.add(reward, game_done)
            
        if game_done:
            self.env.reset()

        self.iterations+= 1
        self.score+= reward
        

        
    def save(self):
        self.model.save(self.save_path)

    def load(self):
        self.model.load(self.save_path)
    
    
    def _calc_q_valuesA(self, rewards, values, done):
        result = numpy.zeros((len(rewards), 1))

        for i in reversed(range(len(rewards)-1)):
            if done[i]:
                gamma = 0.0
            else:
                gamma = self.gamma
            
            result[i][0] = rewards[i] + gamma*values[i+1][0]

        return result

    
    def _calc_q_valuesB(self, rewards, done):
        result = numpy.zeros((len(rewards), 1))
        r  = 0.0

        for i in reversed(range(len(rewards))):
            if done[i]:
                gamma = 0.0
            else:
                gamma = self.gamma
            
            r = rewards[i] + gamma*r
            result[i][0] = r

        return result
    



