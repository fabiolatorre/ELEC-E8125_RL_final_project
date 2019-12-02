import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import transforms
from PIL import Image
import os


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64

        self.reshaped_size = 40*40

        self.fc1 = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc2_ac = torch.nn.Linear(int(self.hidden), action_space)
        self.fc2_cr = torch.nn.Linear(int(self.hidden), 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)  #, -1e-3, 1e-3)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Common part
        x = self.fc1(x)
        x = F.relu(x)

        # Actor part
        x_mean = self.fc2_ac(x)
        x_probs = F.softmax(x_mean, dim=-1)
        dist = Categorical(x_probs)

        # Critic part
        value = self.fc2_cr(x)

        return dist, value


class Agent(object):
    def __init__(self, policy, player_id=1):
        # self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_device = torch.device("cpu")
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-3)
        self.player_id = player_id
        self.gamma = 0.98
        # self.policy.eval()  # uncomment if testing

        self.states = []
        self.action_probs = []
        self.rewards = []
        self.values = []

        self.stacked_obs = None
        self.prev_stacked_obs = None

        # Previous observations (greatest number = oldest) - better using a FIFO stack
        self.obs_prev_1 = 0
        # self.obs_prev_2 = 0
        # self.obs_prev_3 = 0

    def reset(self):
        self.stacked_obs = None
        self.prev_stacked_obs = None
        pass

    def get_name(self):
        return "Agent Smith"

    def load_model(self):
        try:
            weights = torch.load("../models/model_0.mdl", map_location=torch.device('cpu'))
            self.policy.load_state_dict(weights, strict=False)
        except FileNotFoundError:
            print("Model not found. Check the path and try again.")

    def save_model(self, output_directory, episode=0):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        f_name = "model_{}.mdl".format(episode)
        torch.save(self.policy.state_dict(), f_name)

    def episode_finished(self):
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        values = torch.stack(self.values, dim=0).to(self.train_device).squeeze(-1)

        self.states, self.action_probs, self.rewards, self.values = [], [], [], []

        # Compute discounted rewards
        discounted_rewards = discount_rewards(rewards, self.gamma)

        # Compute advantage
        advantage = discounted_rewards - values

        # Normalize advantage
        advantage -= torch.mean(advantage)
        advantage /= torch.std(advantage.detach())

        weighted_probs = - action_probs * advantage.detach()

        # Compute actor and critic loss
        actor_loss = weighted_probs.mean()
        critic_loss = advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss

        ac_loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

    def store_outcome(self, action_prob, action, reward):
        self.states.append(self.prev_stacked_obs)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = self.preprocess_no_fade(observation).to(self.train_device)
        dist, value = self.policy.forward(x)

        if evaluation:
            action = torch.argmax(dist.probs)  # take most probable action
        else:
            action = dist.sample()  # sample from distribution

        act_log_prob = dist.log_prob(action)

        self.values.append(value)

        return action, act_log_prob

    def preprocess(self, observation):
        # Image scaling
        observation = observation[::5, ::5].mean(axis=-1)
        backgroung_threshold = 50
        threshold = 40
        n_past_steps = 3
        ball_color = 255

        # Thresholding
        observation = np.where(observation < backgroung_threshold, 0, observation)  # delete background
        observation = np.where(np.logical_and(observation < ball_color, observation >= backgroung_threshold), threshold,
                               observation)  # set bars to threshold
        observation = np.where(observation == ball_color, n_past_steps * threshold, observation)  # set latest bal value

        if self.stacked_obs is None:
            self.stacked_obs = observation
        else:
            threshold_indices = self.stacked_obs >= threshold
            self.stacked_obs[threshold_indices] -= threshold
            self.stacked_obs = self.stacked_obs + observation

            self.stacked_obs = np.where(self.stacked_obs > n_past_steps * threshold, n_past_steps * threshold,
                                        self.stacked_obs)  # handle overlappings

        return torch.from_numpy(self.stacked_obs).float().flatten()

    def preprocess_no_fade(self, observation):
        # Image scaling
        observation = Image.fromarray(observation)
        observation = transforms.Grayscale()(observation)
        observation = transforms.Resize((40, 40), 0)(observation)
        observation = transforms.ToTensor()(observation)
        observation = observation.squeeze().numpy()

        # Set parameters
        background_threshold = 0.2
        bars_color = 0.5
        shrink_term = 0.8

        # Apply thresholds
        observation = np.where(observation < background_threshold, 0, observation)  # delete background
        observation = np.where(np.logical_and(observation > 0, observation < 1), bars_color, observation)

        # Compute stacked_obs
        if self.stacked_obs is None:
            self.stacked_obs = observation
        else:
            self.prev_stacked_obs = self.stacked_obs
            # self.stacked_obs = observation + shrink_term * (self.obs_prev_1 + self.obs_prev_2 + self.obs_prev_3)
            self.stacked_obs = observation + shrink_term * self.obs_prev_1

        # TODO: consider using stack
        # Update previous observations
        # self.obs_prev_3 = self.obs_prev_2  # oldest
        # self.obs_prev_2 = self.obs_prev_1
        self.obs_prev_1 = np.where(observation == bars_color, 0, observation)

        return torch.from_numpy(self.stacked_obs).float().flatten()
