import gym
import numpy
import torch


gym.envs.register(
    id='MountainCarCustom-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=4096      # MountainCar-v0 uses 200
)

env = gym.make("MountainCarCustom-v0")


class SetRewardRange(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward < 0:
            reward = -0.001

        if done: 
            reward = 1.0
        
        return obs, reward, [done, done], info


env = SetRewardRange(env)
env.reset()

obs             = env.observation_space
actions_count   = env.action_space.n


class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        neurons_count = 64
        self.features_layers = [ 
                                torch.nn.Linear(input_shape[0], neurons_count),
                                torch.nn.ReLU(),
                                ]

        self.policy_layers = [
                                torch.nn.Linear(neurons_count, neurons_count),
                                torch.nn.ReLU(),
                                torch.nn.Linear(neurons_count, outputs_count)
                            ]

        self.value_layers = [
                                torch.nn.Linear(neurons_count, neurons_count),
                                torch.nn.ReLU(),
                                torch.nn.Linear(neurons_count, 1)
                            ]

        self.features_model = torch.nn.Sequential(*self.features_layers)
        self.policy_model   = torch.nn.Sequential(*self.policy_layers)
        self.value_model    = torch.nn.Sequential(*self.value_layers)


    def forward(self, state):  
        features = self.features_model.forward(state)   
        return self.policy_model.forward(features), self.value_model.forward(features)



model  = Model(obs.shape, actions_count)

observation = env.reset()

iterations          = 0
score               = 0.0

observations_batch  = []
actions_batch       = []
rewards_batch       = []
dones_batch         = []

def compute_q_vals(rewards_v, dones_v, gamma = 0.99):

        result = numpy.zeros(len(rewards_v))

        for i in reversed(range(len(rewards_v) - 1)):
            if dones_v[i]:
                gamma_ = 0.0
            else:
                gamma_ = gamma

            result[i] = rewards_v[i] + gamma_*result[i+1]

        return result

optimizer  = torch.optim.Adam(model.parameters(), lr= 0.01)

iterations_now  = 0
iterations_prev = 0
k = 0.9

moves_to_win = 4096.0

while iterations < 1000000:

    observation_t = torch.FloatTensor(observation)
    logits, _ = model.forward(observation_t)

    probs = torch.nn.functional.softmax(logits).detach().numpy()
    action = numpy.random.choice(len(probs), p=probs)
        
    observation, reward, done, _ = env.step(action)

    round_done = done[0]
    game_done  = done[1] 

    observations_batch.append(observation.copy())
    actions_batch.append(action)
    rewards_batch.append(reward)
    dones_batch.append(round_done)

    iterations+= 1
    score+= reward

    if reward > 0.0:
        iterations_prev = iterations_now
        iterations_now  = iterations
        moves_to_win = k*moves_to_win + (1.0 - k)*(iterations_now - iterations_prev)

    
    if iterations%100 == 0:
        env.render()
        print(iterations, score, moves_to_win)
    
    if round_done:
        optimizer.zero_grad()

        observations_v = torch.FloatTensor(observations_batch)
        logits_v, value_v = model.forward(observations_v)

        values_target_v = compute_q_vals(rewards_batch, dones_batch)
        values_target_v = torch.FloatTensor(values_target_v)

        value_loss     = ((values_target_v - value_v.squeeze(-1))**2).mean()

        log_probs_v = torch.nn.functional.log_softmax(logits_v, dim = 1)
        actions_v   = torch.LongTensor(actions_batch)
        log_probs_actions_v = value_v.squeeze(-1).detach()*log_probs_v[range(len(observations_v)), actions_v]
        loss_policy         = -log_probs_actions_v.mean()
        
        prob_v             = torch.nn.functional.softmax(logits_v, dim = 1)
        loss_entropy_v     = 0.1*(prob_v*log_probs_v).sum(dim = 1).mean()

        loss = value_loss  + loss_policy + loss_entropy_v

        loss.backward()
        optimizer.step()

        print("\n\n\n")
        print("value_loss  = ", value_loss.detach().numpy())
        print("loss_policy = ", loss_policy.detach().numpy())
        print("loss_entropy_v = ", loss_entropy_v.detach().numpy())
        print("\n\n\n")

        observations_batch  = []
        actions_batch       = []
        rewards_batch       = []
        dones_batch         = []

    if round_done:
        observation = env.reset()


'''
while agent.iterations < 1000000:
    agent.main()

    if agent.iterations%100 == 0:
        env.render()
        print(agent.iterations, agent.score)
'''