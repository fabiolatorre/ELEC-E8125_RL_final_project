import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import random
from PIL import Image
#from utils.utils import preprocess_ppo_cnn
import copy

MODEL_EPISODE = 2200


class Policy(torch.nn.Module):
    def __init__(self, action_space=3, device=torch.device("cuda")):
        super(Policy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.ep_clip = 0.1
        self.hidden_neurons = 512
        self.reshaped_size = 3520

        self.conv1 = torch.nn.Conv2d(2, 16, 8, 4)
        self.conv2 = torch.nn.Conv2d(16, 32, 4, 2)
        self.fc1 = torch.nn.Linear(3520, 256)
        self.fc2 = torch.nn.Linear(256, self.action_space)

        # self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if type(m) is torch.nn.Linear:
    #             torch.nn.init.normal_(m.weight)  #, -1e-3, 1e-3)
    #             torch.nn.init.zeros_(m.bias)

    def layers(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)

    def forward(self, obs_batch, action_batch, action_prob_batch, advantage_batch):

        vs = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        # vs = np.array([[1., 0.], [0., 1.]])

        ts = torch.FloatTensor(vs[action_batch.cpu().numpy()]).to(self.device)

        logits = self.layers(obs_batch)
        r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob_batch
        loss1 = r * advantage_batch
        loss2 = torch.clamp(r, 1 - self.ep_clip, 1 + self.ep_clip) * advantage_batch
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)

        return loss


class Agent(object):
    def __init__(self, player_id=1, evaluation=True, device_name="cuda"):
        self.name = "Agent Smith"
        self.device = torch.device(device_name)
        self.policy = Policy(device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.player_id = player_id
        self.gamma = 0.99

        self.evaluation = evaluation
        if self.evaluation:
            self.policy.eval()

        self.previous_observation = None

    def reset(self):
        self.previous_observation = None

    def get_name(self):
        return self.name

    def load_model(self):
        try:
            weights = torch.load("../models/model_"+str(MODEL_EPISODE)+".mdl", map_location=torch.device(self.device))
            self.policy.load_state_dict(weights, strict=False)
        except FileNotFoundError:
            print("Model not found. Check the path and try again.")

    def save_model(self, output_directory, episode=0):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        f_name = "{}/model_{}.mdl".format(output_directory, episode)
        torch.save(self.policy.state_dict(), f_name)

    def get_action(self, observation):
        with torch.no_grad():
            observation = self.preprocess(observation)
            logits = self.policy.layers(observation)
            if self.evaluation:
                action = int(torch.argmax(logits[0]).detach().cpu().numpy())
                return action
            else:
                c = torch.distributions.Categorical(logits=logits)
                action = int(c.sample().cpu().numpy()[0])
                action_prob = float(c.probs[0, action].detach().cpu().numpy())
                return action, action_prob, observation

    def episode_batch_finished(self, obs_history, action_history, action_prob_history, reward_history):
        # Compute discounted rewards
        R = 0
        discounted_rewards = []

        for r in reward_history[::-1]:
            if r != 0: R = 0
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

        for _ in range(5):
            n_batch = int(0.7 * len(action_history))
            idxs = random.sample(range(len(action_history)), n_batch)
            obs_batch = torch.cat([obs_history[idx] for idx in idxs], 0).to(self.device)
            action_batch = torch.LongTensor([action_history[idx] for idx in idxs]).to(self.device)
            action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs]).to(self.device)
            advantage_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs]).to(self.device)

            self.optimizer.zero_grad()
            loss = self.policy(obs_batch, action_batch, action_prob_batch, advantage_batch)
            loss.backward()
            self.optimizer.step()

    def preprocess(self, observation):
        """ prepro 200x200x3 uint8 frame into 6000 (75x80) 1D float vector """
        observation = observation[:, 8:192]  # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        observation = observation[::2, ::2] # downsample by factor of 2.
        observation[observation == 58] = 0  # erase background (background type 1)
        observation[observation == 43] = 0  # erase background (background type 2)
        observation[observation == 48] = 0  # erase background (background type 3)
        observation[observation != 0] = 255  # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively

        # img = Image.fromarray(observation, 'RGB')
        # # img.save('my.png')
        # img.show()
        # observation = torch.from_numpy(observation.astype(np.float32).ravel()).unsqueeze(0)
        observation = observation[::, ::].mean(axis=-1)

        if self.previous_observation is None:
            self.previous_observation = copy.deepcopy(observation[np.newaxis, np.newaxis, :, :])

        observation = observation[np.newaxis, np.newaxis, :, :]
        stack_ob = np.concatenate((observation, self.previous_observation), axis=1)
        self.previous_observation = copy.deepcopy(observation)
        return torch.tensor(stack_ob, device=self.device, dtype=torch.float)
