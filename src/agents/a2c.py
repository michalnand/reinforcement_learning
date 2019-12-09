import numpy
import torch

import agents.agent_stats
import common.buffer_a2c

class Agent():
    def __init__(self, env, model, config, save_path = None, save_stats = True):
        self.env = env
        self.save_path = save_path
 
        self.gamma          = config.gamma
        self.entropy_beta   = config.entropy_beta
        self.batch_size     = config.batch_size
       
        self.observation_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        self.buffer = common.buffer_a2c.Buffer(self.batch_size)

        self.model      = model.Model(self.observation_shape, self.actions_count)

        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.observation    = env.reset()
        self.enable_training()

        self.iterations = 0

        self.score      = 0

        if save_path != None and save_stats:
            self.training_stats = agents.agent_stats.AgentStats(self.save_path + "result/training")
            self.testing_stats  = agents.agent_stats.AgentStats(self.save_path + "result/testing")

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self):              
        policy  = self.model.get_policy(self.observation)

        '''
        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
        '''

        probs = self.softmax(policy)
        action = numpy.random.choice(len(probs), p=probs)
        
        self.observation, reward, done, _ = self.env.step(action)

        round_done = done[0]
        game_done  = done[1] 
 

        if self.enabled_training:
            if self.buffer.is_full():
                self.train_model()
                self.buffer.clear()
            else:
                self.buffer.add(self.observation, action, reward, round_done, self.model.device)


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


    def softmax(self, x):
        e_x = numpy.exp(x - numpy.max(x))
        return e_x/e_x.sum()


    def compute_q_vals(self, rewards, done, device):

        result = numpy.zeros(len(rewards))
        
        for n in range(len(rewards) - 1):

            if done[n]:
                gamma = 0.0
            else:
                gamma = self.gamma

            result[n] = rewards[n] + result[n + 1]*gamma

        result = result - result.mean()
        result = result/(result.std() + 0.00001)
        
        result = torch.FloatTensor(result).to(device)
        result = result.reshape(len(rewards), 1)
        return result




    def train_model(self):
        self.optimizer.zero_grad()

        states_v, actions_v, rewards_v, done_v = self.buffer.get(self.model.device)

        logits_v, values_v = self.model(states_v)

        values_target_v = self.compute_q_vals(rewards_v, done_v, self.model.device)

        '''
        compute critic loss, as MSE : L = T - V(s)
        '''
        loss_value_v    = torch.nn.functional.mse_loss(values_v, values_target_v)
        critic_loss     = (values_target_v - values_v).pow(2).mean()

        '''
        compute actor loss, L = log(pi(s, a))*(T - V(s))
        log softmax is better for numerical stability
        '''
        log_prob_v          = torch.nn.functional.log_softmax(logits_v, dim = 1)
        advantage_v         = values_target_v - values_v.detach()
        log_prob_actions_v  = advantage_v*log_prob_v[range(self.batch_size), actions_v]
        loss_policy_v       = -log_prob_actions_v.mean()


        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s))
        '''
        prob_v             = torch.nn.functional.softmax(log_prob_v, dim = 1)
        loss_entropy_v     = self.entropy_beta*(prob_v*log_prob_v).sum(dim = 1).mean()



        loss_v = loss_policy_v + loss_entropy_v + loss_value_v
        loss_v.backward()

        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step() 

        print("\n\n")
        print("loss_value_v = ", loss_value_v.detach().cpu().numpy())
        print("loss_policy_v = ", loss_policy_v.detach().cpu().numpy())
        print("loss_entropy_v = ", loss_entropy_v.detach().cpu().numpy())
        print("\n\n")