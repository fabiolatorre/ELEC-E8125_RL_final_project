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


class Value(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.reshaped_size = state_space
        self.fc1 = torch.nn.Linear(self.reshaped_size, self.hidden)
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
        self.reshaped_size = state_space
        self.fc1 = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc2_ac = torch.nn.Linear(int(self.hidden), action_space)
        # self.sigma = torch.nn.Parameter(torch.tensor([10.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x_mean = self.fc2_ac(x)
        x_probs = F.softmax(x_mean, dim=-1)
        dist = Categorical(x_probs)
        return dist



class Agent(object):
    def __init__(self, policy, value, player_id=1):
        # self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_device = torch.device("cpu")
        self.policy = policy.to(self.train_device)
        self.value = value.to(self.train_device)
        self.p_optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.v_optimizer = torch.optim.RMSprop(value.parameters(), lr=5e-3)
        self.player_id = player_id
        self.gamma = 0.98
        self.batch_size = 1
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.values = []
        self.done = []

        # self.stacked_obs = None
        # self.prev_stacked_obs = None

        # Previous observations (greatest number = oldest) - better using a FIFO stack
        # self.obs_prev_1 = 0
        # self.obs_prev_2 = 0
        # self.obs_prev_3 = 0

    def reset(self):  # TODO: needed?
        # self.stacked_obs = None
        # self.prev_stacked_obs = None
        pass

    def get_name(self):
        return "Agent Smith"

    def load_model(self):
        cwd = os.getcwd()
        print(cwd)
        eps= 8100
        weights_p = torch.load("./models/model_p_episode_"+str(eps)+".mdl", map_location=torch.device('cpu'))
        weights_v = torch.load("./models/model_v_episode_"+str(eps)+".mdl", map_location=torch.device('cpu'))

        self.policy.load_state_dict(weights_p, strict=False)
        self.value.load_state_dict(weights_v, strict=False)

    def save_model(self, episode=0, final=False):
        # cwd = os.getcwd()
        # print(cwd)
        if not final:
            f_name_p = "./models/model_p_episode_{}.mdl".format(episode)
            f_name_v = "./models/model_v_episode_{}.mdl".format(episode)
        else:
            f_name_p = "./models/model_p_final.mdl"
            f_name_v = "./models/model_v_final.mdl"
        torch.save(self.policy.state_dict(), f_name_p)
        torch.save(self.value.state_dict(), f_name_v)

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        values = torch.stack(self.values, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards, self.values = [], [], [], []

        #  Compute discounted rewards (use the discount_rewards function)
        discounted_rewards = discount_rewards(rewards, self.gamma)
        error = discounted_rewards - values
        error -= torch.mean(error)
        error /= torch.std(error.detach())
        #  Compute critic loss and advantages
        self.v_optimizer.zero_grad()
        self.p_optimizer.zero_grad()
        #  Compute the optimization term
        weighted_probs = -action_probs * error.detach()
        p_loss = weighted_probs.sum()
        v_loss = error.pow(2).mean()
        #  Compute the gradients of loss w.r.t. network parameters
        p_loss.backward()
        v_loss.backward()
        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.v_optimizer.step()
        self.p_optimizer.step()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        dist = self.policy.forward(x)

        if evaluation:
            action = torch.argmax(dist.probs)  # take most probable action
        else:
            action = dist.sample()  # sample from distribution

        act_log_prob = dist.log_prob(action)
        self.values.append(self.value.forward(x))
        
        return action, act_log_prob

    def store_outcome(self, state, next_state, action_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.states.append(torch.from_numpy(next_state).float())
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

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

        # Update previous observations
        # self.obs_prev_3 = self.obs_prev_2  # oldest
        # self.obs_prev_2 = self.obs_prev_1
        self.obs_prev_1 = np.where(observation == bars_color, 0, observation)

        return torch.from_numpy(self.stacked_obs).float().flatten()
