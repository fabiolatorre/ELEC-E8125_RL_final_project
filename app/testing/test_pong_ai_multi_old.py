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
import agent_smith
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 100000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)
# player = app.SimpleAi(env, player_id)
player = agent_smith.Agent(player_id)

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
        action1 = player.get_action(ob1)
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
            observation = env.reset()
    wr_array.append(win1 / (i + 1))
    wr_array_avg.append(np.mean(wr_array[max(0, len(wr_array)-100):]))
    if i % 20 == 0:
        print("episode {} over. Broken WR: {:.3f}".format(i, wr_array[-1]))
    if i % 5000 == 0:
        player.save_model(str(i))
        print("Model saved")
        plt.plot(wr_array)
        plt.plot(wr_array_avg)
        plt.legend(["WR", "100-episode average"])
        plt.title("WR history")
        plt.savefig('./plots/WR_history_{}_training.pdf'.format(i))

plt.plot(wr_array)
plt.plot(wr_array_avg)
plt.legend(["WR", "100-episode average"])
plt.title("WR history")
plt.savefig('./plots/WR_history_final.pdf')

player.save_model("final")
