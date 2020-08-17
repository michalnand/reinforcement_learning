import torch

def Agent:
    def __init__(self, env, model, config, save_path = None, save_stats = True):
        self.env = env
        self.save_path = save_path
        self.save_stats = save_stats

        self.batch_size = 128

        self.gamma       = 0.99
        self.soft_tau    = 0.01

        hidden_dim = 64

        value_lr  = 0.0003
        soft_q_lr = 0.0003
        policy_lr = 0.0003

        replay_buffer_size = 16384

        self.value_net          = model.ValueNetwork(state_dim, hidden_dim)
        self.target_value_net   = model.ValueNetwork(state_dim, hidden_dim)

        self.soft_q_net1        = model.SoftQNetwork(state_dim, output_dim, hidden_dim)
        self.soft_q_net2        = model.SoftQNetwork(state_dim, output_dim, hidden_dim)

        self.policy_net         = model.PolicyNetwork(state_dim, output_dim, hidden_dim)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer     = torch.optim.Adam(self.value_net.parameters(),    lr=value_lr)
        self.soft_q_optimizer1   = torch.optim.Adam(self.soft_q_net1.parameters(),  lr=soft_q_lr)
        self.soft_q_optimizer2   = torch.optim.Adam(self.soft_q_net2.parameters(),  lr=soft_q_lr)
        self.policy_optimizer    = torch.optim.Adam(self.policy_net.parameters(),   lr=policy_lr)

        self.replay_buffer = ReplayBuffer(replay_buffer_size) 

        self.state = self.env.reset()

    def main(self):
        action = self.policy_network.get_action(self.state)
            
        self.state, reward, done, _ = self.env.step(action)

        if done:
            self.state = self.env.reset()

         if self.enabled_training:
            self.experience_replay.add(self.state, reward, done)
            self.train()


    def train(self):
        states_t, actions_t, rewards_t, next_states_t, dones_t = self.replay_buffer.sample(self.batch_size, self.device)
        

        predicted_q_value1 = self.soft_q_net1(states_t, actions_t)
        predicted_q_value2 = self.soft_q_net2(states_t, actions_t)
        
        predicted_value    = self.value_net(states_t)

        new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(states_t)

        #train soft Q net
        target_value = self.target_value_net(next_state)
        
        target_q_value = rewards_t + (1 - dones_t)*self.gamma*target_value
        q_value_loss1 = (target_q_value.detach() - predicted_q_value1)**2).mean()
        q_value_loss2 = (target_q_value.detach() - predicted_q_value2)**2).mean()


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()    

      
        #train value net
        predicted_new_q_value = torch.min(self.soft_q_net1(states_t, new_action), self.soft_q_net2(states_t, new_action))

        target_value    = predicted_new_q_value - log_prob
        value_loss      = (target_value - predicted_value)**2).mean()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        #train policy net
        policy_loss = (log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        #polyak for the target value network
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_((1.0 - self.soft_tau)*target_param.data + self.soft_tau*param.data)