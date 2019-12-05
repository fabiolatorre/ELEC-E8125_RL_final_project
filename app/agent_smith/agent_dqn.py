import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import transforms
from PIL import Image
import os

MODEL_NUMBER = 100


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(torch.nn.Module):
    def __initn__(self):
        super(DQN, self).__init__()
        self.num
        self.gamma = 0.99

        self.hidden_neurons = 512
        self.reshaped_size = 3520

        self.conv1 = torch.nn.Conv2d(2, 16, 8, 4)
        self.conv2 = torch.nn.Conv2d(16, 32, 4, 2)
        self.fc1 = torch.nn.Linear(3520, 256)
        self.fc2 = torch.nn.Linear(256, self.action_space)


class Agent(object):
    def __init__(self, player_id=1):
        pass

    def reset(self):
        pass

    def get_name(self):
        return "Agent Smith"

    def load_model(self):
        try:
            weights = torch.load("../models/model_"+MODEL_NUMBER+".mdl", map_location=torch.device('cpu'))
            self.policy.load_state_dict(weights, strict=False)
        except FileNotFoundError:
            print("Model not found. Check the path and try again.")

    def save_model(self, output_directory, episode=0):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        f_name = "{}/model_{}.mdl".format(output_directory, episode)
        torch.save(self.policy.state_dict(), f_name)

    def episode_finished(self):
        pass

    def get_action(self, observation, evaluation=False):
        pass

    def preprocess(self, observation):
        pass




import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from utils.utils_dqn import Transition, ReplayMemory



class Agent(object):
    def __init__(self, state_space,
                 batch_size=32, hidden_size=12, gamma=0.98):
        self.n_actions = 3
        self.state_space_dim = state_space
        self.policy_net = DQN(state_space, self.n_actions, hidden_size)
        self.target_net = DQN(state_space, self.n_actions, hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayMemory(12000)
        self.batch_size = batch_size




    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = 1-torch.tensor(batch.done, dtype=torch.uint8)
        non_final_next_states = [s for nonfinal,s in zip(non_final_mask,
                                     batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states)
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Task 4: TODO: Compute the expected Q values
        expected_state_action_values = 0

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(),
                                expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def get_action(self, state, epsilon=0.05):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = torch.from_numpy(state).float()
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.n_actions)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(next_state).float()
        state = torch.from_numpy(state).float()
        self.memory.push(state, action, next_state, reward, done)

