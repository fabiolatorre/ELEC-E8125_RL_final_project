import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r



class Value(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, action_space)
        self.fc3 = torch.nn.Linear(action_space, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.sigma = torch.nn.Parameter(torch.tensor([10.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2_mean(x)
        sigma = self.sigma
        # TODO: Instantiate and return a normal distribution with mean mu and std of sigma (T1)
        return Normal(mu, sigma)

        # TODO: Add a layer for state value calculation (T3)


class Agent(object):
    def __init__(self, policy, value):
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.train_device)
        self.value = value.to(self.train_device)
        self.p_optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.v_optimizer = torch.optim.RMSprop(value.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.values = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        values = torch.stack(self.values, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards, self.values = [], [], [], []

        # TODO: Compute discounted rewards (use the discount_rewards function)
        discounted_rewards = discount_rewards(rewards, self.gamma)
        error = discounted_rewards - values
        error -= torch.mean(error)
        error /= torch.std(error.detach())


        # TODO: Compute critic loss and advantages (T3)
        self.v_optimizer.zero_grad()
        self.p_optimizer.zero_grad()
        # TODO: Compute the optimization term (T1, T3)
        weighted_probs = -action_probs * error.detach()
        p_loss = weighted_probs.sum()
        v_loss = error.pow(2).mean()
        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        p_loss.backward()
        v_loss.backward()
        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.v_optimizer.step()
        self.p_optimizer.step()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1)
        distr = self.policy.forward(x)
        # TODO: Return mean if evaluation, else sample from the distribution returned by the policy (T1)
        if (evaluation):
            action = torch.mean(distr)
        else:
            action = distr.sample()

        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = distr.log_prob(action)
        # TODO: Return state value prediction, and/or save it somewhere (T3)
        self.values.append(self.value.forward(x))
        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
