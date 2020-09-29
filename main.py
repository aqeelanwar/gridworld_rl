"""
Name: gridworld_rl
Description: Tabular Reinforcement Learning using policy iteration (REINFORCE) on gridworld problem
Author: Aqeel Anwar (aqeel.anwar@gatech.edu)

Maze visualization: https://morvanzhou.github.io/tutorials/
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
from maze_env import Maze
from RL_brain import ThetaLearningTable
import numpy as np
import imageio
from matplotlib import rc
from PIL import Image
import os
import datetime
import platform

# Set up plotting parameters
rc("text", usetex=True)
rc("font", family="cmr10")
plt.interactive(True)
line_color = "#94B6D2"
title_font = {"size": "24"}
axis_font = {"size": "12"}

MAZE_H = 10
MAZE_W = 10
MAX_ITERS = 2000

# Define the coordinates for wall and goal positions
HELL_COORD = [
    [[2, 2], [7, 7]],
    [[9, 3], [7, 8]],
    # [[2, 2], [8, 5]],
    # [[2, 7], [8, 5]],
    # [[2, 7], [5, 0]],
    # [[7, 7], [5, 0]],
    # [[0, 6], [9, 6]],
    # [[9, 13], [7, 8]],
    # [[4, 9], [12, 9]],
    # [[6, 12], [12, 6]],
]

GOAL_COORD = [
    [7, 2],
    [5, 4],
    # [5, 8],
    # [8, 1],
    # [2, 4],
    # [2, 9],
    # [13, 5],
    # [5, 13],
    # [12, 3],
    # [6, 14],
]
ORIGIN_COORD = [0, 0]


num_agents = len(HELL_COORD)
img = [[] for _ in range(num_agents)]

# Set learning parameters
num_episode = 1000
num_save_images = 3
env_list = []
Theta_RL_list = []
moving_average_constant = 30
GAMMA = 0.95
epi_r = np.zeros(num_agents)
reward_max_iters = -0.2

cum_r = [[] for _ in range(num_agents)]
mov_avg = [[] for _ in range(num_agents)]
wins = [[] for _ in range(num_agents)]
win_avg = [[] for _ in range(num_agents)]
percentile_25 = [[] for _ in range(num_agents)]
percentile_75 = [[] for _ in range(num_agents)]

if __name__ == "__main__":
    for n in range(num_agents):
        name = "Maze " + str(n)
        hell_coord = HELL_COORD[n]
        goal_coord = GOAL_COORD[n]

        # Generate the maze of size MAZE_H times MAZE_W with goal and wall positions as defined by user
        env = Maze(
            name=name,
            state_nature="SONAR",
            MAZE_H=MAZE_H,
            MAZE_W=MAZE_W,
            UNIT=40,
            hell_coord=hell_coord,
            goal_coord=goal_coord,
            origin_coord=ORIGIN_COORD,
            opposite_goal=False,
        )

        # Set up the tabular parameter for RL
        Theta_RL = ThetaLearningTable(env.n_actions, env.n_states, name=name)
        env_list.append(env)
        Theta_RL_list.append(Theta_RL)

    # env_list and Theta_RL_list contains the maze handle and tabular parameter of all the mazes
    state_action_track = np.zeros((env.n_states, env.n_actions))
    axs = [[] for _ in range(num_agents)]
    figs = [[] for _ in range(num_agents)]
    text_handle = [[] for _ in range(num_agents)]
    for i in range(num_agents):

        # Set up parameters for the output plots
        figs[i], axs[i] = plt.subplots(2, figsize=(6, 8))
        name_fig = "Maze " + str(i) + " "
        axs[i][0].set_xlabel("Number of episodes", fontsize=12)
        axs[i][0].set_ylabel("Cumulative return", fontsize=12)
        axs[i][1].set_xlabel("Number of episodes", fontsize=12)
        axs[i][1].set_ylabel("Win %age", fontsize=12)

    # Generate subplots
    fig_table, axs_table = plt.subplots(num_agents, figsize=(20, 1.2 * num_agents))
    for i, ax in enumerate(axs_table):
        name = "Maze " + str(i)
        ax.title.set_text(name)
    fig_table.suptitle(r"Tabular parameter $\theta(s,a)$", fontsize=16)
    fig_table.text(0.5, 0.04, "States (s)", ha="center", fontdict=axis_font)
    fig_table.text(
        0.08, 0.5, "Actions (a)", va="center", rotation="vertical", fontdict=axis_font
    )

    data = [[] for _ in range(num_agents)]

    # Carry out Monte Carlo estimation of sum of discounted return and update the tabular parameter
    for episode in tqdm(range(num_episode)):
        for i, (env, Theta_RL) in enumerate(zip(env_list, Theta_RL_list)):
            data_tuple = []
            state, observation = env.reset()
            while True:
                # refresh env
                env.render()
                # RL choose action based on observation
                action = Theta_RL.choose_action(state)

                # RL take action and get next observation and reward
                state_, observation_, reward, done, win = env.step(action)
                # Save the tuple
                data_tuple.append([state, action, state_, reward, done])

                state_action_track[state][action] += 1
                # swap observation
                state = state_

                epi_r[i] += reward

                if episode > num_episode - num_save_images:
                    f = "data/f" + str(episode) + "_" + str(len(data_tuple)) + ".png"
                    env.canvas.postscript(file="a_.eps")
                    # use PIL to convert to PNG
                    img[i].append(np.asarray(Image.open("a_.eps")))

                episode_len = len(data_tuple)
                if done or episode_len >= MAX_ITERS:
                    if episode_len >= MAX_ITERS:
                        data_tuple[-1][3] = reward_max_iters
                        data_tuple[-1][4] = True

                    wins[i].append(win)
                    G = np.zeros(episode_len)
                    r = 0
                    for epi in range(episode_len - 1, -1, -1):
                        reward = data_tuple[epi][3]
                        r = reward + r * GAMMA
                        G[epi] = r
                    # Update the parameters
                    Theta_RL.learn(
                        data_tuple, G, episode, num_episode, state_action_track
                    )
                    Theta_table = Theta_RL.theta_table.copy()
                    cum_r[i].append(epi_r[i])
                    epi_r[i] = 0

                    # Plot Theta
                    axs_table[i].cla()
                    axs_table[i].imshow(
                        Theta_table.T, cmap="hot", interpolation="nearest"
                    )

                    name = "Maze " + str(i)
                    axs_table[i].set_title(name, size=14)

                    # Update the output plots when necessary
                    if len(cum_r[i]) > moving_average_constant:
                        # Clear the current plot
                        axs[i][0].cla()
                        axs[i][1].cla()

                        mov_avg[i].append(
                            np.mean(cum_r[i][-moving_average_constant:-1])
                        )
                        win_avg[i].append(np.mean(wins[i][-moving_average_constant:-1]))

                        # Calculate percentiles
                        percentile_25[i].append(
                            np.percentile(cum_r[i][-moving_average_constant:-1], 25)
                        )
                        percentile_75[i].append(
                            np.percentile(cum_r[i][-moving_average_constant:-1], 75)
                        )
                        axs[i][0].plot(
                            np.arange(len(mov_avg[i])) + moving_average_constant,
                            mov_avg[i],
                            color=line_color,
                        )
                        axs[i][1].plot(
                            np.arange(len(mov_avg[i])) + moving_average_constant,
                            win_avg[i],
                            color=line_color,
                        )
                        axs[i][1].set_ylim([-0.1, 1.1])
                        # axs[i].fill_between(np.arange(0, len(percentile_25[i])), percentile_25[i], percentile_75[i], color = line_color, alpha=0.2)

                        # Set up aesthetics for the plot
                        if len(cum_r[i]) > moving_average_constant + 1:
                            text_handle[i].remove()
                        area_str = "Convergence: " + str(np.round(np.mean(wins[i]), 2))
                        text_handle[i] = axs[i][1].text(
                            moving_average_constant,
                            0.9,
                            area_str,
                            style="italic",
                            bbox={"facecolor": "red", "alpha": 0.5},
                            fontsize=12,
                        )
                        name = "Maze " + str(i)
                        # axs[i].title.set_text(name)
                        axs[i][0].set_title(name, size=14)
                        axs[i][0].set_xlabel("Number of episodes", fontsize=12)
                        axs[i][0].set_ylabel("Cumulative return", fontsize=12)

                        axs[i][0].fill_between(
                            np.arange(len(mov_avg[i]))+moving_average_constant,
                            percentile_75[i],
                            percentile_25[i],
                            alpha=0.2,
                            color="blue",
                        )
                        axs[i][1].fill_between(
                            np.arange(len(win_avg[i]))+moving_average_constant,
                            win_avg[i],
                            0,
                            alpha=0.2,
                            color="blue",
                        )
                        axs[i][1].set_xlabel("Number of episodes", fontsize=12)
                        axs[i][1].set_ylabel("Win Percentage", fontsize=12)
                        # plt.subplots_adjust(hspace=0.5)
                        plt.show()

                    # break while loop when end of this episode
                    break

    # Create animated GIF of the last few frames for visualization
    print("Creating gif")
    platform = platform.node()
    x = str(datetime.datetime.now())
    dt = x.replace("-", "").replace(":", "").replace(".", "").replace(" ", "_")[:-6]
    folder_name = "Results_" + dt
    os.mkdir(folder_name)

    for i in range(len(env_list)):
        name_init = folder_name + os.path.sep + "Maze_" + str(i)
        Theta_RL_list[i].save_theta(path=name_init)
        name_mov_avg = name_init + "_mov_avg.npy"
        np.save(name_mov_avg, mov_avg[i])
        name_win_avg = name_init + "_win_avg.npy"
        np.save(name_win_avg, win_avg[i])
        name = name_init + ".gif"
        print(name)
        imageio.mimsave(name, img[i], duration=0.05)
        print("Area under curve: ", np.mean(cum_r[i]))
        print("---------------------------------------")
        # Save the graphs
        name = folder_name + os.path.sep + "Maze_" + str(i) + "_return.pdf"
        figs[i].savefig(name, bbox_inches="tight")
    theta_filepath = folder_name + os.path.sep + "Theta.pdf"
    fig_table.savefig(theta_filepath, bbox_inches="tight")
    print("Done")
