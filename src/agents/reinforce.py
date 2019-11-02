import numpy
import torch

import agents.agent_stats

import common.policy_state_value_replay


import common.atari_wrapper


 
class Agent():
    def __init__(self, env, model, config, save_path = None, save_stats = True):
        self.env = env
        self.save_path = save_path

        self.action = 0

        self.batch_size     = config.batch_size

        self.gamma          = config.gamma

        self.experience_replay =  common.policy_state_value_replay.Buffer(config.experience_replay_size)

        self.observation_shape = self.env.observation_space.shape
        self.actions_count     = self.env.env.action_space.n

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
        logits  = self.model.get_q_values(self.observation)
        probs   = self.softmax(logits)
        self.action = self.choose_action(probs)

        observation_new, self.reward, self.done, self.info = self.env.step(self.action)

        if self.enabled_training:
            if self.experience_replay.is_full() == False:
                self.experience_replay.add(self.observation, self.action, self.reward, self.done)
            else:   
                self.train_model()

        self.observation = observation_new

        if hasattr(self, "training_stats") and hasattr(self, "testing_stats"):
            if self.enabled_training:
                self.training_stats.add(self.reward, self.done)
            else:
                self.testing_stats.add(self.reward, self.done)
            
            
        if self.done:
            self.env.reset()

        self.iterations+= 1
        self.score+= self.reward
        
        
    def train_model(self):
        self.experience_replay.compute(self.gamma)
        

        batches_count = self.experience_replay.length()//self.batch_size

        for _ in range(0, batches_count):
            observation, state_value, actions = self.experience_replay.get_random_batch(self.batch_size, self.model.device)
            
            output = self.model.forward(observation)
            probs = torch.softmax(output, dim = 1)
            log_prob_v = torch.log_softmax(output, dim=1)
            log_prob_actions_v = state_value*log_prob_v[range(self.batch_size), actions]

            policy_loss = -log_prob_actions_v.mean()
            
            entropy = -(probs*torch.log(probs)).sum(dim = 1)
            entropy_loss = -0.01*entropy.mean()


            loss   = policy_loss + entropy_loss
    
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-10.0, 10.0)
            self.optimizer.step()


        self.experience_replay.clear()                

        
    def softmax(self, logits):
        m = numpy.max(logits)
        l = numpy.exp(logits - m)
        return l/numpy.sum(l)

    def choose_action(self, probs):
        return numpy.random.choice(range(len(probs)), p=probs)
        
    def save(self):
        self.model.save(self.save_path)

    def load(self):
        self.model.load(self.save_path)