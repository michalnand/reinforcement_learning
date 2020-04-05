import numpy
import torch

import agents.agent_stats

from torch.distributions import Categorical

class Agent():
    def __init__(self, envs, model, config, save_path = None, save_stats = True):
        self.envs = envs
        self.save_path = save_path

        self.envs_count = len(self.envs)
 
        self.gamma          = config.gamma
        self.entropy_beta   = config.entropy_beta
        self.batch_size     = config.batch_size
       
        self.observation_shape = self.envs[0].observation_space.shape
        self.action_size       = self.envs[0].action_space.shape[0]

        
        self.model          = model.Model(self.observation_shape, self.action_size)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.observations = []

        for env in self.envs:
            self.observations.append(env.reset())

        self.enable_training()

        self.iterations = 0

        self.score      = 0

        self._init_buffer()

        if save_path != None and save_stats:
            self.training_stats = agents.agent_stats.AgentStats(self.save_path + "result/training")
            self.testing_stats  = agents.agent_stats.AgentStats(self.save_path + "result/testing")


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False


    def _init_buffer(self):
        self.mu_b               = torch.zeros((self.envs_count, self.batch_size, self.action_size)).to(self.model.device)
        self.var_b              = torch.zeros((self.envs_count, self.batch_size, self.action_size)).to(self.model.device)
        self.values_b           = torch.zeros((self.envs_count, self.batch_size, 1)).to(self.model.device)
        self.action_b           = torch.zeros((self.envs_count, self.batch_size, self.action_size))
        self.rewards_b          = numpy.zeros((self.envs_count, self.batch_size))
        self.done_b             = numpy.zeros((self.envs_count, self.batch_size), dtype=bool)

        self.idx = 0

        
    def process_env(self, env_id = 0):
        observation_t      = torch.tensor(self.observations[env_id], dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)
        
        mu, var, value     = self.model.forward(observation_t)

        dist               = torch.distributions.Normal(mu, var)
        action             = dist.sample()[0].detach().clamp(-1.0, 1.0)
        action_np          = action.to("cpu").numpy()

        self.observations[env_id], reward, done, _ = self.envs[env_id].step(action_np)
        
        round_done = done[0]
        game_done  = done[1] 

        if self.enabled_training:
            self.mu_b[env_id][self.idx]         = mu.squeeze(0)
            self.var_b[env_id][self.idx]        = var.squeeze(0)
            self.values_b[env_id][self.idx]     = value.squeeze(0)
            self.action_b[env_id][self.idx]     = action
            self.rewards_b[env_id][self.idx]    = reward
            self.done_b[env_id][self.idx]       = round_done

        if game_done:
            self.envs[env_id].reset()

        if hasattr(self, "training_stats") and hasattr(self, "testing_stats"):
            if env_id == 0:
                if self.enabled_training:
                    self.training_stats.add(reward, game_done)
                else:
                    self.testing_stats.add(reward, game_done)

        return reward
        
    def compute_loss(self, env_id = 0):
        target_values_b = self._calc_q_values(self.rewards_b[env_id], self.values_b[env_id].detach().cpu().numpy(), self.done_b[env_id])

        target_values_b = torch.FloatTensor(target_values_b).to(self.model.device)

       
        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        loss_value = (target_values_b - self.values_b[env_id])**2
        loss_value = loss_value.mean()


        '''
        compute actor loss 
        L = log(pi(s, a, mu, var))*(T - V(s)) = _calc_log_prob(mu, var, action)*A
        '''
        log_probs   = self._calc_log_prob(self.mu_b[env_id], self.var_b[env_id], self.action_b[env_id])
        advantage   = (target_values_b - self.values_b[env_id]).detach()
        loss_policy = -log_probs*advantage
        loss_policy = loss_policy.mean()

        '''
        compute entropy loss, to avoid greedy strategy
        '''
        loss_entropy = -(1.0 + torch.log(2.0*3.141592654*self.var_b[env_id]))*0.5
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        #final loss
        loss = loss_value + loss_policy + loss_entropy
        return loss


    
    def main(self):
        reward = 0
        for env_id in range(self.envs_count):
            tmp = self.process_env(env_id)
            if env_id == 0:
                reward = tmp

        if self.enabled_training:
            self.idx+= 1
        
        if self.idx > self.batch_size-1:      
            loss = 0
            for env_id in range(self.envs_count):
                loss+= self.compute_loss(env_id)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step() 

            #clear batch buffer
            self._init_buffer()

            '''
            print("loss_value = ", loss_value.detach().cpu().numpy())
            print("loss_policy = ", loss_policy.detach().cpu().numpy())
            print("loss_entropy = ", loss_entropy.detach().cpu().numpy())
            print("\n\n\n")
            '''

        self.iterations+= 1
        self.score+= reward

            
    def save(self):
        self.model.save(self.save_path)

    def load(self):
        self.model.load(self.save_path)
    

    def _calc_q_values(self, rewards, critic_value, done):
        size = len(rewards)
        result = numpy.zeros((size, 1))

        q = 0.0
        for n in reversed(range(size)):
            if done[n]:
                gamma = 0.0
            else:
                gamma = self.gamma

            q = rewards[n] + gamma*q
            result[n][0] = q

        return result

    def _calc_log_prob(self, mu, var, action):
        result = -((action - mu)**2) / (2.0*var.clamp(min = 0.001))
        result+= -torch.log(torch.sqrt(2.0*3.141592654*var))

        return result