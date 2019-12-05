import gym
import numpy as np
from matplotlib import pyplot as plt
from agent_smith.agent_dqn import Agent as DQNAgent  # Task 4
from itertools import count
import torch
from utils.utils_dqn import plot_rewards

env_name = "CartPole-v0"
env = gym.make(env_name)
env.reset()

# Values for DQN  (Task 4)
if "CartPole" in env_name:
    TARGET_UPDATE = 20
    glie_a = 200
    num_episodes = 5000
    hidden = 12
    gamma = 0.98
    replay_buffer_size = 50000
    batch_size = 32
elif "LunarLander" in env_name:
    TARGET_UPDATE = 20
    glie_a = 10000
    num_episodes = 15000
    hidden = 64
    gamma = 0.95
    replay_buffer_size = 12000
    batch_size = 128
else:
    raise ValueError("Please provide hyperparameters for %s" % env_name)


# Get number of actions from gym action space
n_actions = env.action_space.n
state_space_dim = env.observation_space.shape[0]

# Task 4 - DQN
agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size,
              hidden, gamma)

# Training loop
cumulative_rewards = []
for ep in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    done = False
    eps = glie_a/(glie_a+ep)
    cum_reward = 0
    while not done:
        # Select and perform an action
        action = agent.get_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        cum_reward += reward

        # Task 4: Update the DQN

        # Move to the next state
        state = next_state
    cumulative_rewards.append(cum_reward)
    plot_rewards(cumulative_rewards)

    # Update the target network, copying all weights and biases in DQN
    # Uncomment for Task 4
    if ep % TARGET_UPDATE == 0:
        agent.update_target_network()

    # Save the policy
    # Uncomment for Task 4
    if ep % 1000 == 0:
        torch.save(agent.policy_net.state_dict(),
                  "weights_%s_%d.mdl" % (env_name, ep))

print('Complete')
plt.ioff()
plt.show()

