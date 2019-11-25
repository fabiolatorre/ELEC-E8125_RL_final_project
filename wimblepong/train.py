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
from agent_smith import Agent, Policy
from PIL import Image

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")

# Number of episodes/games to play
episodes = 100000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)

policy = Policy(3)
player = Agent(player_id)

# Set the names for both players
env.set_names(player.get_name(), opponent.get_name())

# Arrays to keep track of rewards
reward_history = []
average_reward_history = []

win1 = 0
wr_array = []
wr_array_avg = []

for episode in range(0, episodes):
    reward_sum = 0
    player.reset()
    ob1, ob2 = env.reset()
    done = False
    while not done:
        # Get the actions from both players
        action1 = player.get_action(ob1)
        action2 = opponent.get_action()

        prev_ob1 = ob1

        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

        # Count the wins
        if rew1 == 10:
            win1 += 1

        # Give reward for surviving, really helpful at the beginning
        rew1 += 0.01

        # Store action's outcome (so that the agent can improve its policy)
        player.store_outcome(prev_ob1, action1, rew1)

        if done:
            observation = env.reset()

        # Store total episode reward
        reward_sum += rew1

        player.episode_finished()
    wr_array.append(win1 / (episode + 1))
    wr_array_avg.append(np.mean(wr_array[max(0, len(wr_array)-100):]))

    if not episode % 20 and episode:
        print("episode {} over. Broken WR: {:.3f}. AVG reward: {}".format(episode, wr_array[-1], average_reward_history[-1]))

    reward_history.append(reward_sum)
    average_reward_history.append(np.mean(reward_history[max(0, len(reward_history)-100):]))

    if not episode % 100 and episode:
        # Save model during training
        player.save_model()
        print("Model saved")

    if not episode % 1000 and episode:
        # Create plot of the training performance
        plt.plot(wr_array)
        plt.plot(wr_array_avg)
        plt.legend(["WR", "100-episode average"])
        plt.title("WR history")
        plt.savefig('./plots/WR_history_training.pdf')

# Create final plot of the training performance
plt.plot(wr_array)
plt.plot(wr_array_avg)
plt.legend(["WR", "100-episode average"])
plt.title("WR history")
plt.savefig('./plots/WR_history_final.pdf')

# Save final model
player.save_model(final=True)
