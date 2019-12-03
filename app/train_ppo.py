import gym
import numpy as np
from wimblepong import SimpleAi
from agent_smith.agent_ppo import Agent
from utils.utils import plot
import os
import torch

PLOTS_DIR = os.path.abspath("./plots")
MODELS_DIR = os.path.abspath("./models")

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")

# Number of episodes/games to play
episodes = 100000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = SimpleAi(env, opponent_id)

# Get dimensionalities of actions and observations
observation_space_dim = env.observation_space.shape[-1]
action_space_dim = env.action_space.n

player = Agent(player_id)

# Set the names for both players
env.set_names(player.get_name(), opponent.get_name())

# Lists to keep track of rewards
reward_history, reward_history_avg = [], []
wr_array, wr_array_avg = [], []
episode_length_history, episode_length_avg = [], []

# Initialize winning rate
win1 = 0
wr_reset = 1000

for episode in range(episodes):
    if not episode % wr_reset:
        win1 = 0  # Reset WR

    # Initialize values
    reward_sum = 0
    episode_length = 0
    done = False

    # Resets
    player.reset()
    ob1, ob2 = env.reset()

    while not done:
        # Get the actions from both players
        with torch.no_grad():
            action1, action_prob1 = player.get_action(ob1, evaluation=False)
        action2 = opponent.get_action()

        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1+2, action2))

        # Count the wins
        if rew1 == 10:
            win1 += 1

        # Store action's outcome (so that the agent can improve its policy)
        player.store_outcome(action1, action_prob1, rew1)

        # Store total episode reward
        reward_sum += rew1
        episode_length += 1

    if not episode % 10:
        player.episode_finished()

    # Update WR values for plots
    wr_array.append(win1 / ((episode % wr_reset)+1))
    wr_array_avg.append(np.mean(wr_array[max(0, len(wr_array)-100):]))

    # Update reward values for plots
    reward_history.append(reward_sum)
    reward_history_avg.append(np.mean(reward_history[max(0, len(reward_history) - 100):]))

    # Update episode_length values
    episode_length_history.append(episode_length)
    episode_length_avg.append(np.mean(episode_length_history[max(0, len(episode_length_history) - 100):]))

    if not episode % 20 and episode:
        print("Episode {} over. Broken WR: {:.3f}. AVG reward: {:.3f}. Episode legth: {:.2f}."
              .format(episode, wr_array[-1], reward_history_avg[-1], episode_length_avg[-1]))

    if not episode % 100 and episode:
        # Save model during training
        player.save_model(MODELS_DIR, episode)
        print("Model saved")

    if not episode % 1000 and episode:
        # Create plot of the training performance WR
        plot(wr_array, wr_array_avg, "WR history", "WR_history_training",
             PLOTS_DIR, ["WR", "100-episode average"])

        # Create plot of the training performance reward
        plot(reward_history, reward_history_avg, "Reward history", "reward_history_training",
             PLOTS_DIR, ["Reward", "100-episode average"])

# Save final model
player.save_model(MODELS_DIR, episodes)

# Create final plot of the training performance WR
plot(wr_array, wr_array_avg, "WR history", "WR_history_final",
     PLOTS_DIR, ["WR", "100-episode average"])

# Create final plot of the training performance reward
plot(reward_history, reward_history_avg, "Reward history", "reward_history_final",
     PLOTS_DIR, ["Reward", "100-episode average"])
