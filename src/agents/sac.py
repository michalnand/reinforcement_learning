def Agent:
    def __init__(self, env, model, config, save_path = None, save_stats = True):
        self.env = env

        self.batch_size     = 64
        self.device         = ?
        self.gamma          = 0.99
        self.alpha          = 0.2
        self.update_step    = 4

        soft_q_network_lr = 0.002
        policy_network_lr = 0.002
        alpha_lr          = 0.002

        replay_buffer_size = 16384

        #soft Q model
        self.soft_q_network = model.SoftQNetwork(input_dim, output_dim)
        self.soft_q_network_optimizer = torch.optim.Adam(self.soft_q_network.parameters(), lr=soft_q_network_lr)

        #policy model
        self.policy_network = model.PolicyNetwork(input_dim, output_dim)
        self.policy_network_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=policy_network_lr)

        # entropy temperature
        self.target_entropy     = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha          = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer    = torch.optim.Adam([self.log_alpha], lr=alpha_lr)


        #replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.state = self.env.reset()

        self.step = 0

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

        #Q net mse loss
        soft_q_loss = (target_q_t.detach() - predicted_q_t)**2
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

        
        # update temperature
        alpha_loss = (self.log_alpha * (-next_log_pi_t - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        self.step+= 1
        

