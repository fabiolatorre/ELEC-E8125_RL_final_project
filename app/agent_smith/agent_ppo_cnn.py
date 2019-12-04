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
    def __init__(self, action_space=3):
        super(Policy, self).__init__()
        self.num_frames = 2
        self.action_space = action_space
        self.eps_clip = 0.1
        self.hidden_neurons = 512
        self.reshaped_size = 3520
        #self.input_dim = 9300 * 2

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
        # x = x.reshape(-1, self.reshaped_size)
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)


    def forward(self, d_obs, action=None, action_prob=None, advantage=None, deterministic=False):
        if action is None:
            with torch.no_grad():
                logits = self.layers(d_obs)
                if deterministic:
                    action = int(torch.argmax(logits[0]).detach().cuda().numpy())
                    action_prob = 1.0
                else:
                    c = torch.distributions.Categorical(logits=logits)
                    action = int(c.sample().cuda().numpy()[0])
                    action_prob = float(c.probs[0, action].detach().cuda().numpy())
                return action, action_prob

        # # policy gradient (REINFORCE)
        # logits = self.layers(d_obs)
        # loss = F.cross_entropy(logits, action, reduction='none') * advantage
        # return loss.mean()


        # PPO
        if self.action_space == 3:
            vs = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        elif self.action_space == 2:
            vs = np.array([[1., 0.], [0., 1.]])
        ts = torch.FloatTensor(vs[action.cuda().numpy()])

        logits = self.layers(d_obs)
        r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob
        loss1 = r * advantage
        loss2 = torch.clamp(r, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)

        return loss


class Agent(object):
    def __init__(self, player_id=1, evaluation=True):
        # self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_device = torch.device("cuda")
        self.policy = Policy().to(self.train_device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.player_id = player_id
        self.gamma = 0.99

        self.evaluation = evaluation
        if self.evaluation:
            self.policy.eval()

        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []

        self.discounted_rewards = []

        self.pp_observation = None
        self.previous_observation = None

    def reset(self):
        self.previous_observation = None

    def reset_lists(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []

    def get_name(self):
        return "Agent Smith"

    def load_model(self):
        try:
            weights = torch.load("../models/model_"+str(MODEL_EPISODE)+".mdl", map_location=torch.device('cuda'))
            self.policy.load_state_dict(weights, strict=False)
        except FileNotFoundError:
            print("Model not found. Check the path and try again.")

    def save_model(self, output_directory, episode=0):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        f_name = "{}/model_{}.mdl".format(output_directory, episode)
        torch.save(self.policy.state_dict(), f_name)

    def episode_finished(self):
        R = 0
        self.discounted_rewards = []
        for r in self.rewards[::-1]:
            if r != 0: R = 0  # scored/lost a point in pong, so reset reward sum
            R = r + self.gamma * R
            self.discounted_rewards.insert(0, R)

        self.discounted_rewards = torch.FloatTensor(self.discounted_rewards)
        self.discounted_rewards = (self.discounted_rewards - self.discounted_rewards.mean()) / self.discounted_rewards.std()

        self.update_policy()

    def store_outcome(self, action, action_prob, reward):
        self.states.append(self.pp_observation)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)

    def update_policy(self):
        for _ in range(5):
            n_batch = 20000
            idxs = random.sample(range(len(self.actions)), min(len(self.actions), n_batch))
            d_obs_batch = torch.cat([self.states[idx] for idx in idxs], 0)
            action_batch = torch.LongTensor([self.actions[idx] for idx in idxs])
            action_prob_batch = torch.FloatTensor([self.action_probs[idx] for idx in idxs])
            advantage_batch = torch.FloatTensor([self.discounted_rewards[idx] for idx in idxs])
            # advantage_batch = (advantage_batch - advantage_batch.mean()) / advantage_batch.std()

            self.optimizer.zero_grad()
            loss = self.policy.forward(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
            loss.backward()
            self.optimizer.step()





    def preprocess(self, observation):
        """ prepro 200x200x3 uint8 frame into 6000 (75x80) 1D float vector """
        observation = observation[:, 8:192]  # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        observation = observation[::2, ::2] # downsample by factor of 2.
        observation[observation == 58] = 0  # erase background (background type 1)
        observation[observation == 43] = 0  # erase background (background type 2)
        observation[observation == 48] = 0  # erase background (background type 3)
        #observation[observation != 0] = 1  # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively

        # img = Image.fromarray(observation, 'RGB')
        # # img.save('my.png')
        # img.show()
        # observation = torch.from_numpy(observation.astype(np.float32).ravel()).unsqueeze(0)
        observation = observation[::, ::].mean(axis=-1)
        #observation = np.expand_dims(observation, axis=-1)

        if self.previous_observation is None:
            self.previous_observation = copy.deepcopy(observation[np.newaxis, np.newaxis, :, :])

        # rtn = torch.cat([self.previous_observation, observation], dim=1)

        observation = observation[np.newaxis, np.newaxis, :, :]
        stack_ob = np.concatenate((observation, self.previous_observation), axis=1)
        #stack_ob = stack_ob.transpose(1, 3)
        self.previous_observation = copy.deepcopy(observation)
        rtn = torch.tensor(stack_ob, device=self.train_device, dtype=torch.float)
        return rtn

    # def preprocess(self, observation):
    #     observation = observation[::2, ::2].mean(axis=-1)
    #     observation = np.expand_dims(observation, axis=-1)
    #     if self.previous_observation is None:
    #         self.previous_observation = observation
    #     stack_ob = np.concatenate((self.previous_observation, observation), axis=-1)
    #     stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0)
    #     stack_ob = stack_ob.transpose(1, 3)
    #     self.previous_observation = observation
    #     return stack_ob




    def get_action(self, observation):
        self.pp_observation = self.preprocess(observation)
        action, action_prob = self.policy.forward(self.pp_observation, deterministic=self.evaluation)
        if self.evaluation:
            return action
        else:
            return action, action_prob