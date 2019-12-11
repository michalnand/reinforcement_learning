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


        self.model      = model.Model(self.observation_shape, self.actions_count)

        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.observation    = env.reset()
        self.enable_training()

        self.iterations = 0

        self.score      = 0

        if save_path != None and save_stats:
            self.training_stats = agents.agent_stats.AgentStats(self.save_path + "result/training")
            self.testing_stats  = agents.agent_stats.AgentStats(self.save_path + "result/testing")

        self.logits_b           = []
        self.state_values_b     = []
        self.action_b           = []
        self.rewards_b          = []
        self.done_b             = []

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self):              

        observation_t  = torch.tensor(self.observation, dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)
        logits_t, value_t  = self.model.forward(observation_t)


        action_probs_t        = torch.nn.functional.softmax(logits_t, dim=1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
       
            
        self.observation, reward, done, _ = self.env.step(action_t.item())


        
        round_done = done[0]
        game_done  = done[1] 


        if self.enabled_training:
            self.logits_b.append(logits_t.squeeze(0))
            self.state_values_b.append(value_t.squeeze(0)[0])
            self.action_b.append(action_t.item())
            self.rewards_b.append(reward)
            self.done_b.append(done)

 
        if len(self.logits_b) >= self.batch_size:

            size = len(self.logits_b)

            target_value_t = torch.FloatTensor(size)

            v = 0.0
            for n in reversed(range(size)):
                if self.done_b:
                    gamma = 0.0
                else:
                    gamma = self.gamma

                v = self.rewards_b[n] + gamma*v
                target_value_t[n] = v

            target_value_t.to(self.model.device)
            

            loss_value  = 0
            loss_policy = 0
            loss_entropy= 0
            for logits, value, action, target_value in zip(self.logits_b, self.state_values_b, self.action_b, target_value_t):
                advantage       = target_value  - value.item()

                probs     = torch.nn.functional.softmax(logits, dim = 0)
                log_probs = torch.nn.functional.log_softmax(logits, dim = 0)

                '''
                compute critic loss, as MSE : L = (T - V(s))^2
                '''
                loss_value+= (target_value - value)**2.0

                '''
                compute actor loss, L = log(pi(s, a))*(T - V(s)) = log(pi(s, a))*A
                TODO : log softmax is better for numerical stability
                '''
                loss_policy+= -log_probs[action]*advantage

                '''
                compute entropy loss, to avoid greedy strategy
                L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
                '''
                loss_entropy+= self.entropy_beta*(probs*log_probs).sum()

            loss_value.to(self.model.device)
            loss_policy.to(self.model.device)
            loss_entropy.to(self.model.device)

            '''
            print("\n\n")
            print("loss_value_v   = ", loss_value.detach().cpu().numpy())
            print("loss_policy_v  = ", loss_policy.detach().cpu().numpy())
            print("loss_entropy_v = ", loss_entropy.detach().cpu().numpy())
            print("\n\n")
            '''

            loss = loss_value + loss_policy  + loss_entropy
            loss.to(self.model.device)

            loss.backward()

            for param in self.model.parameters():
                param.grad.data.clamp_(-10.0, 10.0)
            self.optimizer.step() 

            self.optimizer.zero_grad()

 
            self.logits_b           = []
            self.state_values_b     = []
            self.action_b           = []
            self.rewards_b          = []
            self.done_b             = []



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




