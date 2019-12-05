import gym
import numpy as np
from wimblepong import SimpleAi
from agent_smith.agent_dqn import Agent
from utils.utils import plot
import os
import torch

PLOTS_DIR = os.path.abspath("./plots")
MODELS_DIR = os.path.abspath("./models")

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = SimpleAi(env, opponent_id)

player = Agent(player_id, evaluation=False, device_name="cpu")

# Set the names for both players
env.set_names(player.get_name(), opponent.get_name())

# Lists to keep track of rewards
reward_history, reward_history_avg = [], []
wr_array, wr_array_avg = [], []
episode_length_history, episode_length_avg = [], []

# Initialize winning rate
win1 = 0
wr_reset = 1000

target_update = 8
glie_a = 5000

episode = 0

# Training loop
cumulative_rewards = []
while True:
    # Initialize values
    reward_sum = 0
    episode_length = 0
    done = False
    eps = glie_a/(glie_a+episode)
    eps = max(eps, 0.03)

    # Resets
    player.reset()
    ob1, _ = env.reset()

    ob1 = player.preprocess(ob1)

    if not episode % wr_reset:
        win1 = 0  # Reset WR

    while not done:
        # Get the actions from both players

        action1 = player.get_action_train(ob1, epsilon=eps)
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (next_ob1, _), (rew1, _), done, _ = env.step((action1, action2))
        next_ob1 = player.preprocess(next_ob1)

        # Count the wins
        if rew1 == 10:
            win1 += 1

        rew1 /= 10

        player.store_transition(ob1, action1, next_ob1, rew1, done)
        if not episode_length % 6:
            player.update_network()

        # Move to the next state
        ob1 = next_ob1

        # Store total episode reward
        reward_sum += rew1
        episode_length += 1

    if not episode % target_update:
        player.update_target_network()

    # ---PLOTTING AND SAVING MODEL---

    # Update WR values for plots
    wr_array.append(win1 / ((episode % wr_reset) + 1))
    wr_array_avg.append(np.mean(wr_array[max(0, len(wr_array) - 100):]))

    # Update reward values for plots
    reward_history.append(reward_sum)
    reward_history_avg.append(np.mean(reward_history[max(0, len(reward_history) - 100):]))

    # Update episode_length values
    episode_length_history.append(episode_length)
    episode_length_avg.append(np.mean(episode_length_history[max(0, len(episode_length_history) - 100):]))

    if not episode % 20 and episode:
        print("Episode {} over. Broken WR: {:.3f}. AVG reward: {:.3f}. Episode legth: {:.2f}."
              .format(episode, wr_array[-1], reward_history_avg[-1], episode_length_avg[-1]))

    if not episode % 1000 and episode:
        # Save model
        player.save_model(MODELS_DIR, episode)
        print("Model saved")

        # Update plot of the Winning Rate
        plot(wr_array, wr_array_avg, "WR history", "WR_history_training",
             PLOTS_DIR, ["WR", "100-episode average"])

        # Update plot of the reward
        plot(reward_history, reward_history_avg, "Reward history", "reward_history_training",
             PLOTS_DIR, ["Reward", "100-episode average"])

        # Update plot of the episode length
        plot(episode_length_history, episode_length_avg, "Episode length", "episode_length_training",
             PLOTS_DIR, ["Episode length", "100-episode average"])

    episode += 1
