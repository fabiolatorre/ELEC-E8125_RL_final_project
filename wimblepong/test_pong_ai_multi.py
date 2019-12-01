"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from agent_smith import Agent, Policy, Value
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
# env = gym.make("WimblepongVisualMultiplayer-v0")
env = gym.make("WimblepongMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 100000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)
# player = wimblepong.SimpleAi(env, player_id)

# Get dimensionalities of actions and observations
observation_space_dim = env.observation_space.shape[-1]
action_space_dim = env.action_space.n

policy = Policy(observation_space_dim, action_space_dim)
value = Value(observation_space_dim, action_space_dim)
player = Agent(policy, value, player_id)
player.load_model()

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

win1 = 0
wr_array = []
wr_array_avg = []
for i in range(0, episodes):
    player.reset()
    ob1, ob2 = env.reset()
    done = False
    while not done:
        # Get the actions from both SimpleAIs
        action1, action_prob1 = player.get_action(ob1)
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        # img = Image.fromarray(ob1)
        # img.save("ob1.png")
        # img = Image.fromarray(ob2)
        # img.save("ob2.png")
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
        if done:
            observation= env.reset()
            print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
