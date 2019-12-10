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

        self.probs_b            = []
        self.logprobs_b         = []
        self.state_values_b     = []
        self.rewards_b          = []
        self.done_b             = []

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self):              

        observation_t  = torch.tensor(self.observation, dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)
        policy_t, value_t  = self.model.forward(observation_t)


        action_probs_t        = torch.nn.functional.softmax(policy_t, dim=1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
       
            
        self.observation, reward, done, _ = self.env.step(action_t.item())


        
        round_done = done[0]
        game_done  = done[1] 


        if self.enabled_training:
            self.probs_b.append(action_probs_t)
            self.logprobs_b.append(action_distribution_t.log_prob(action_t))
            self.state_values_b.append(value_t)
            self.rewards_b.append(reward)
            self.done_b.append(done)

 
        if len(self.probs_b) >= self.batch_size:
            self.optimizer.zero_grad()

            dis_reward = 0
            target_value_t = []
            for reward in self.rewards_b[::-1]:
                dis_reward = reward + self.gamma * dis_reward
                target_value_t.insert(0, dis_reward)
                
        
            loss = 0

            loss_value  = 0
            loss_policy = 0
            loss_entropy= 0
            for prob, logprob, value, target_value in zip(self.probs_b, self.logprobs_b, self.state_values_b, target_value_t):
                advantage       = target_value  - value.item()


                '''
                compute critic loss, as MSE : L = (T - V(s))^2
                '''
                loss_value+= (target_value - value)**2 

                '''
                compute actor loss, L = log(pi(s, a))*(T - V(s)) = log(pi(s, a))*A
                TODO : log softmax is better for numerical stability
                '''
                loss_policy+= -logprob * advantage

                '''
                compute entropy loss, to avoid greedy strategy
                L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
                '''
                loss_entropy+= self.entropy_beta*(prob*logprob).sum()

                


            print("\n\n")
            print("loss_value_v = ", loss_value.detach().cpu().numpy())
            print("loss_policy_v = ", loss_policy.detach().cpu().numpy())
            print("loss_entropy_v = ", loss_entropy.detach().cpu().numpy())
            print("\n\n")


            loss = loss_value + loss_policy + loss_entropy
            

            loss.backward()

            for param in self.model.parameters():
                param.grad.data.clamp_(-10.0, 10.0)
            self.optimizer.step() 


            self.probs_b            = []
            self.logprobs_b         = []
            self.state_values_b     = []
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




