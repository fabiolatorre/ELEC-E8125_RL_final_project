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
import torch

# Make the environment
env = gym.make("WimblepongMultiplayer-v0")

# Number of episodes/games to play
episodes = 100000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)

a = env.observation_space

policy = Policy(env.observation_space.shape[-1], env.action_space.n)
player = Agent(policy, player_id)

# Set the names for both players
env.set_names(player.get_name(), opponent.get_name())

# Arrays to keep track of rewards
reward_history = []
average_reward_history = []

win1 = 0
wr_array = []
wr_array_avg = []

episode_length_history = []
average_episode_length = []

for episode in range(0, episodes):
    reward_sum = 0
    player.reset()
    ob1, ob2 = env.reset()
    done = False
    episode_length = 0
    while not done:
        # Get the actions from both players
        action1, action_prob1 = player.get_action(ob1)
        action2 = opponent.get_action()

        # prev_ob1 = ob1

        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1.detach(), action2))

        # Count the wins
        if rew1 == 10:
            win1 += 1

        # Give reward for surviving
        # rew1 += 0.1

        # Store action's outcome (so that the agent can improve its policy)
        player.store_outcome(ob1, action_prob1, action1, rew1)

        if done:
            player.store_next_values(torch.tensor([0.0]))  # save also the values of the next state
        else:
            x = torch.from_numpy(ob1).float().to(player.train_device)
            _, v_next = player.policy.forward(x)
            player.store_next_values(v_next)
        # Store total episode reward
        reward_sum += rew1
        episode_length += 1

    player.episode_finished()

    # Update WR values for plots
    wr_array.append(win1 / (episode + 1))
    wr_array_avg.append(np.mean(wr_array[max(0, len(wr_array)-100):]))

    # Update reward values for plots
    reward_history.append(reward_sum)
    average_reward_history.append(np.mean(reward_history[max(0, len(reward_history) - 100):]))

    # Update episode_length values
    episode_length_history.append(episode_length)
    average_episode_length.append(np.mean(episode_length_history[max(0, len(episode_length_history) - 100):]))

    if not episode % 20 and episode:
        print("Episode {} over. Broken WR: {:.3f}. AVG reward: {}. Episode legth: {:.2f}.".format(episode, wr_array[-1], average_reward_history[-1], average_episode_length[-1]))

    if not episode % 100 and episode:
        # Save model during training
        player.save_model(episode)
        print("Model saved")

    if not episode % 1000 and episode:
        # Create plot of the training performance WR
        plt.plot(wr_array)
        plt.plot(wr_array_avg)
        plt.legend(["WR", "100-episode average"], loc='upper left')
        plt.title("WR history")
        plt.savefig('./plots/WR_history_training.pdf')
        plt.clf()

        # Create plot of the training performance reward
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"], loc='upper left')
        plt.title("Reward history")
        plt.savefig('./plots/reward_history_training.pdf')
        plt.clf()

# Save final model
player.save_model(final=True)

# Create final plot of the training performance
plt.plot(wr_array)
plt.plot(wr_array_avg)
plt.legend(["WR", "100-episode average"], loc='upper left')
plt.title("WR history")
plt.savefig('./plots/WR_history_final.pdf')
plt.clf()

# Create final plot of the training performance
plt.plot(reward_history)
plt.plot(average_reward_history)
plt.legend(["reward", "100-episode average"], loc='upper left')
plt.title("Reward history")
plt.savefig('./plots/reward_history_final.pdf')
