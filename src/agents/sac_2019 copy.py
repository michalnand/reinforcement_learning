def Agent:
    def __init__(self, env, model, config, save_path = None, save_stats = True):
        self.env = env

        self.batch_size = 32
        self.device     = ?

        soft_q_network_lr = 0.002
        policy_network_lr = 0.002

        replay_buffer_size = 16384

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.soft_q_network = model.SoftQNetwork(input_dim, output_dim)
        self.policy_network = model.PolicyNetwork(input_dim, output_dim)

        self.soft_q_network_optimizer = optim.Adam(self.soft_q_network.parameters(), lr=soft_q_network_lr)
        self.policy_network_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_network_lr)

        self.state = self.env.reset()

    def main(self):
        action, _ = self.policy_network(self.state)
            
        self.state, reward, done, _ = self.env.step(action)

        if done:
            self.state = self.env.reset()

         if self.enabled_training:
            self.experience_replay.add(self.state, reward, done)
            self.train()

    def train(self):
        states_t, actions_t, rewards_t, next_states_t, dones_t = self.replay_buffer.sample(self.batch_size, self.device)
        
        next_actions_t, next_log_pi_t = self.policy_net.sample(next_states_t)
        
        #q values, regularized with policy
        next_q_t = self.soft_q_network(next_states_t, next_actions_t) - self.alpha*next_log_pi
        target_q_t = rewards_t + (1 - dones_t)*self.gamma*next_q_t

        #prediction
        predicted_q_t = self.soft_q_network(states_t, actions_t)

        #q net mse loss
        soft_q_loss = target_q_t.detach() - predicted_q_t)**2
        soft_q_loss = soft_q_loss.mean()
        
        #train q-net
        self.soft_q_network_optimizer.zero_grad()
        soft_q_loss.backward()
        self.soft_q_network_optimizer.step()

        #update policy net
        if self.step%self.update_step == 0:
            new_actions, log_std = self.policy_net.sample(states_t)
            q_values = self.soft_q_network(states_t, new_actions)
            
            policy_loss = self.alpha*log_std - q_values.detach()
            policy_loss = policy_loss.mean()


            #train policy-net
            self.soft_q_network_optimizer.zero_grad()
            soft_q_loss.backward()
            self.soft_q_network_optimizer.step()
        


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from models import SoftQNetwork, PolicyNetwork
from common.replay_buffers import BasicBuffer





class SACAgent:
  
    def __init__(self, env, gamma, tau, alpha, q_lr, policy_lr, a_lr, buffer_maxlen):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.action_range = [env.action_space.low, env.action_space.high]
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2
        
        # initialize networks 
        self.q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.policy_net = PolicyNetwork(self.obs_dim, self.action_dim).to(self.device)

        # copy params to target param
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param)

        # initialize optimizers 
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # entropy temperature
        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)

        self.replay_buffer = BasicBuffer(buffer_maxlen)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()


        
        
        return self.rescale_action(action)
    
    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
            (self.action_range[1] + self.action_range[0]) / 2.0
   
    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)
        
        next_actions, next_log_pi = self.policy_net.sample(next_states)
        next_q1 = self.target_q_net1(next_states, next_actions)
        next_q2 = self.target_q_net2(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        # q loss
        curr_q1 = self.q_net1.forward(states, actions)
        curr_q2 = self.q_net2.forward(states, actions)        
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # update q networks        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # delayed update for policy network and target q networks
        new_actions, log_pi = self.policy_net.sample(states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(
                self.q_net1.forward(states, new_actions),
                self.q_net2.forward(states, new_actions)
            )
            policy_loss = (self.alpha * log_pi - min_q).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
            # target networks
            for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # update temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.update_step += 1