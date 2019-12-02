import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import transforms
from PIL import Image


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

        self.reshaped_size = state_space

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
        self.batch_size = 1
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.done = []
        # self.policy.eval()  # uncomment if testing

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
        weights = torch.load("../models/model_training.mdl", map_location=torch.device('cpu'))
        self.policy.load_state_dict(weights, strict=False)

    def save_model(self, episode=0, final=False):
        if not final:
            f_name = "./models/model_episode_{}.mdl".format(episode)
        else:
            f_name = "./models/model_final.mdl"
        torch.save(self.policy.state_dict(), f_name)

    def episode_finished(self, episode_number):
        all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        all_states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        all_next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        all_done = torch.Tensor(self.done).to(self.train_device)
        self.states, self.next_states, self.actions, self.rewards, self.done = [], [], [], [], []

        # Compute discounted rewards
        discounted_rewards = discount_rewards(all_rewards, self.gamma)

        # Compute state value estimates
        _, old_state_values = self.policy(all_states)
        _, next_state_values = self.policy(all_next_states)

        # Zero out values of terminal states
        next_state_values = next_state_values.squeeze(-1) * (1 - all_done)

        # Detach, squeeze, etc.
        next_state_values = next_state_values.detach()
        old_state_values = old_state_values.squeeze(-1)

        # Estimate of state value and critic loss
        updated_state_values = discounted_rewards + self.gamma * next_state_values
        critic_loss = F.mse_loss(old_state_values, updated_state_values.detach())

        # # Compute advantage
        # advantage = discounted_rewards - old_state_values

        # Estimate advantage
        advantages = updated_state_values - old_state_values
        # advantages -= torch.mean(advantages)
        # advantages /= torch.std(advantages.detach())

        # Weighted log probs
        weighted_probs = - all_actions * advantages.detach()

        # Compute actor and critic loss
        actor_loss = torch.mean(weighted_probs)  # TODO: check sign
        loss = actor_loss + critic_loss

        loss.backward()

        if (episode_number+1) % self.batch_size == 0:
            self.update_policy()

        return critic_loss.item()

    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        dist, _ = self.policy.forward(x)

        if evaluation:
            action = torch.argmax(dist.probs)  # take most probable action
        else:
            action = dist.sample()  # sample from distribution

        act_log_prob = dist.log_prob(action)

        return action, act_log_prob

    def store_outcome(self, state, next_state, action_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.actions.append(action_prob)
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
