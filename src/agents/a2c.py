import numpy
import torch

import agents.agent_stats
import common.buffer_a2c

class Agent():
    def __init__(self, env, model, config, save_path = None, save_stats = True):
        self.env = env
        self.save_path = save_path

        self.action = 0

        self.gamma    = config.gamma
        self.entropy  = config.entropy
        self.update_rate = config.update_rate
       
        self.observation_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        self.buffer = common.buffer_a2c.Buffer(self.update_rate)

        self.model      = model.Model(self.observation_shape, self.actions_count)

        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.observation    = env.reset()
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
        policy_dist, value  = self.model.get_output(self.observation)

        action = numpy.random.choice(len(policy_dist), p=policy_dist)
        
        observation_new, self.reward, done, self.info = self.env.step(action)

        round_done = done[0]
        game_done  = done[1]
 
        if self.enabled_training:
            if self.buffer.is_full():
                self.train_model()
                self.buffer.clear()
            else:
                self.buffer.add(self.observation, self.action, self.reward, round_done)
 
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
        

        
    def save(self):
        self.model.save(self.save_path)

    def load(self):
        self.model.load(self.save_path)

    def train_model(self):
        observations, actions, rewards, done = self.buffer.get(self.model.device)

        policy_dists, values  = self.model.forward(observations)

        log_probs = []
        for n in range(self.buffer.length()):
            action = actions[n]
            log_prob = torch.log(policy_dists[n].squeeze(0)[action])
            log_probs.append(log_prob)

        q_values = numpy.zeros(values.shape)
        q_val = 0.0
        for n in reversed(range(len(rewards))):
            q_val = rewards[n] + self.gamma*q_val
            q_values[n] = q_val 

        
        #TODO - divergence
        
        q_values  = torch.FloatTensor(q_values).to(self.model.device)
        log_probs = torch.stack(log_probs)
        
        
        advantage = q_values - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss # + 0.001 * entropy_term

        print("loss = ", actor_loss, critic_loss, "\n\n")

        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()
        






'''
def a2c(env):
    num_steps = 300
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    
    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        for steps in range(num_steps):
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy() 

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state
            
            if done or steps == num_steps-1:
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0,0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:                    
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                break
        
        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
  
        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()
'''