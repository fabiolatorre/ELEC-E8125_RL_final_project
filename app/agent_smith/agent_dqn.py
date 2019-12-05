import numpy as np
import torch
import torch.nn.functional as F
import os
from collections import namedtuple
import random
import copy

MODEL_EPISODE = 2200

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


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
    def __init__(self, action_space=3, device=torch.device("cuda")):
        super(DQN, self).__init__()
        self.device = device
        self.hidden_neurons = 256
        self.reshaped_size = 3520
        self.action_space = action_space

        self.conv1 = torch.nn.Conv2d(4, 16, 8, 4)
        self.conv2 = torch.nn.Conv2d(16, 32, 4, 2)
        self.fc1 = torch.nn.Linear(3520, 256)
        self.fc2 = torch.nn.Linear(256, self.action_space)

        # self.init_weights()

    def init_weights(self):
        try:
            weights = torch.load("./models/model_" + str(MODEL_EPISODE) + ".mdl",
                                 map_location=torch.device(self.device))
            self.load_state_dict(weights, strict=False)
        except FileNotFoundError:
            print("Model not found. Check the path and try again.")

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        flat_size = x.size()[0]
        x = x.view(flat_size, -1)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Agent(object):
    def __init__(self, player_id=1, evaluation=True, device_name="cuda"):
        self.name = "Agent Smith"
        self.device = torch.device(device_name)
        self.n_actions = 3
        self.policy_net = DQN(device=self.device).to(self.device)
        self.target_net = DQN(device=self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay_buffer_size = 60000  # TODO: tune
        self.memory = ReplayMemory(self.replay_buffer_size)
        self.batch_size = 120   # TODO: tune
        self.gamma = 0.99  # TODO: tune
        self.target_net.eval()  # TODO: check

        self.evaluation = evaluation
        if self.evaluation:
            self.policy_net.eval()

        self.previous_observation = None

    def get_name(self):
        return "Agent Smith"

    def load_model(self):
        try:
            weights = torch.load("../models/model_" + str(MODEL_EPISODE) + ".mdl", map_location=torch.device('cpu'))
            self.policy_net.load_state_dict(weights, strict=False)
        except FileNotFoundError:
            print("Model not found. Check the path and try again.")

    def save_model(self, output_directory, episode=0):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        f_name = "{}/model_{}.mdl".format(output_directory, episode)
        torch.save(self.policy_net.state_dict(), f_name)

    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset(self):
        self.previous_observation = []

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
        non_final_next_states = torch.stack(non_final_next_states).squeeze(1).to(self.device)
        state_batch = torch.stack(batch.state).squeeze(1).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask.bool()] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(),
                                expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def get_action_train(self, state, epsilon=0.05):
        # state = self.preprocess(state)
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                # state = torch.from_numpy(state).float()
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.n_actions)

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        # next_state = torch.from_numpy(next_state).float()
        # state = torch.from_numpy(state).float()
        self.memory.push(state, action, next_state, reward, done)

    def preprocess(self, observation):
        """ prepro 200x200x3 uint8 frame into 6000 (75x80) 1D float vector """
        observation = observation[:,
                      8:192]  # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        observation = observation[::2, ::2]  # downsample by factor of 2.
        observation[observation == 58] = 0  # erase background (background type 1)
        observation[observation == 43] = 0  # erase background (background type 2)
        observation[observation == 48] = 0  # erase background (background type 3)
        observation[
            observation != 0] = 255  # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively

        # img = Image.fromarray(observation, 'RGB')
        # # img.save('my.png')
        # img.show()
        # observation = torch.from_numpy(observation.astype(np.float32).ravel()).unsqueeze(0)
        observation = observation[::, ::].mean(axis=-1)

        if len(self.previous_observation) == 0:
            self.previous_observation.append(np.copy(observation[np.newaxis, np.newaxis, :, :]))
            self.previous_observation.append(np.copy(observation[np.newaxis, np.newaxis, :, :]))
            self.previous_observation.append(np.copy(observation[np.newaxis, np.newaxis, :, :]))

        observation = observation[np.newaxis, np.newaxis, :, :]
        stack_ob = np.concatenate(
            (observation, self.previous_observation[0], self.previous_observation[1], self.previous_observation[2]),
            axis=1)
        self.previous_observation[2] = np.copy(self.previous_observation[1])
        self.previous_observation[1] = np.copy(self.previous_observation[0])
        self.previous_observation[0] = np.copy(observation)
        rtn = torch.tensor(stack_ob, device=self.device, dtype=torch.float)
        return rtn


