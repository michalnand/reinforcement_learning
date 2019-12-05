import numpy
import torch

import agents.agent_stats
import common.buffer_a2c

class Agent():
    def __init__(self, env, model, config, save_path = None, save_stats = True):
        self.env = env
        self.save_path = save_path
 
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
        policy, value  = self.model.get_output(self.observation)

        probs = self.softmax(policy)
        action = numpy.random.choice(len(probs), p=probs)
        
        self.observation, reward, done, _ = self.env.step(action)

        round_done = done[0]
        game_done  = done[1] 
 

        if self.enabled_training:
            if self.buffer.is_full() or round_done:
                self.train_model()
                self.buffer.clear()
            else:
                self.buffer.add(self.observation, action, reward, round_done)


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

    def compute_q_vals(self, rewards, device):
        result = numpy.zeros(len(rewards), dtype=numpy.float32)

        sum = 0.0
        for i in reversed(range(len(rewards))):
            sum*= self.gamma
            sum+= rewards[i]
            result[i] = sum

        result = torch.from_numpy(result).to(self.model.device)
        return result

    def actions_one_hot(self, actions):
        actions_one_hot = torch.zeros((len(actions), self.actions_count), requires_grad=False)

        for i in range(len(actions)):
            actions_one_hot[i][actions[i]] = 1.0

        return actions_one_hot.to(self.model.device)

    def compute_entropy(self, logits):
        b = torch.nn.functional.softmax(logits, dim=1) * torch.nn.functional.log_softmax(logits, dim=1)
        b = -1.0*b.mean()
        return b

    def train_model(self):
        states, actions, rewards, done = self.buffer.get(self.model.device)

        q_vals = self.compute_q_vals(rewards, self.model.device)
                
        actor, critic = self.model(states)

        log_prob = torch.nn.functional.log_softmax(actor, dim = 1)

        advantage = (q_vals - critic)
        log_prob_actions = advantage*log_prob[range(len(states)), actions]

        loss_actor      = -log_prob_actions.mean()
        loss_critic     = (advantage**2).mean()
        loss_entropy    = -0.1*self.compute_entropy(actor)

        

        self.optimizer.zero_grad()
        loss = loss_actor + loss_critic + loss_entropy
        loss.backward()
        self.optimizer.step()

        print(loss_actor.detach().cpu().numpy(), loss_critic.detach().cpu().numpy(), loss_entropy.detach().cpu().numpy(), "\n\n\n")








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