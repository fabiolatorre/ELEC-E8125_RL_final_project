import torch
import numpy as np
from torch.distributions import Categorical
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyConv(torch.nn.Module):
    def __init__(self, action_space, hidden=64):
        super().__init__()
        self.action_space = action_space
        self.hidden = hidden
        self.conv1 = torch.nn.Conv2d(2, 32, 3, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 2)
        self.reshaped_size = 128*11*11
        self.fc1_actor = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc1_critic = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.fc2_value = torch.nn.Linear(self.hidden, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = x.reshape(-1, self.reshaped_size)
        x_ac = self.fc1_actor(x)
        x_ac = F.relu(x_ac)
        x_mean = self.fc2_mean(x_ac)

        x_probs = F.softmax(x_mean, dim=-1)
        dist = Categorical(x_probs)

        x_cr = self.fc1_critic(x)
        x_cr = F.relu(x_cr)
        value = self.fc2_value(x_cr)

        return dist, value



class Agent(object):
    def __init__(self, player_id=1):
        self.train_device = "cpu"       # CPU
        # self.train_device = "cuda"    # GPU
        self.policy = PolicyConv(3, 128).to(self.train_device)
        self.prev_obs = None
        self.player_id = player_id
        self.policy.eval()

    def replace_policy(self):
        self.old_policy.load_state_dict(self.policy.state_dict())

    def get_action(self, observation):
        x = self.preprocess(observation).to(self.train_device)
        dist, value = self.policy.forward(x)
        action = torch.argmax(dist.probs)
        return action

    def reset(self):
        self.prev_obs = None

    def get_name(self):
        return "Agent Smith"

    def load_model(self):
        weights = torch.load("model.mdl", map_location=torch.device('cpu'))     # CPU
        # weights = torch.load("model.mdl")                                     # GPU
        self.policy.load_state_dict(weights, strict=False)

    def save_model(self, ep):
        f_name = 'model_'+ ep + ".mdl"
        torch.save(self.policy.state_dict(), f_name)

    def preprocess(self, observation):
        observation = observation[::2, ::2].mean(axis=-1)
        observation = np.expand_dims(observation, axis=-1)
        if self.prev_obs is None:
            self.prev_obs = observation
        stack_ob = np.concatenate((self.prev_obs, observation), axis=-1)
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0)
        stack_ob = stack_ob.transpose(1, 3)
        self.prev_obs = observation
        return stack_ob

