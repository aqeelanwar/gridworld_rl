"""
Name: gridworld_rl
Description: Tabular Reinforcement Learning using policy iteration (REINFORCE) on gridworld problem
Author: Aqeel Anwar (aqeel.anwar@gatech.edu)

Maze visualization: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class ThetaLearningTable:
    def __init__(self, num_actions, num_states, learning_rate=0.2, name='', init='uniform'):
        self.name = name
        self.num_actions = num_actions
        self.num_states = num_states
        self.lr = learning_rate
        if init=='uniform':
            self.theta_table = 0.5 * np.ones((num_states, num_actions), dtype=np.float64)
        elif init=='random':
            self.theta_table = np.random.rand(num_states, num_actions)

    def update_with_grad(self, grad):
        self.theta_table = self.theta_table + self.lr * grad

    def assign(self, theta):
        self.theta_table = theta.copy()

    def load(self, path='model/theta.npy'):
        self.theta_table = np.load(path)

    def save_theta(self, path):
        path = path + '_theta.npy'
        np.save(path, self.theta_table)

    def choose_action(self, state):
        num_actions = self.num_actions
        probs_raw = self.theta_table[state]
        probs = softmax(probs_raw)
        action = np.random.choice(num_actions, 1, p=probs)[0]
        return action

    def pi_s(self, state):
        probs_raw = self.theta_table[state]
        probs = softmax(probs_raw)
        return probs

    # def pi(self, state):
    #     probs_raw = self.theta_table[state]
    #     probs = softmax(probs_raw)
    #     return probs

    def learn(self, data_tuple, G, episode, num_episode, state_action_track):

        # Exponential decay
        # alpha_k = np.exp(-a * episode / num_episode)

        # Linear decay
        # alpha_k = 1 - episode/num_episode

        # No decay
        alpha_k = 1
        # print(alpha_k)
        for epi, data in enumerate(data_tuple):
            # if G[epi]==1:
            #     self.lr = 2
            #     print('double')
            # else:
            #     self.lr = 0.2
            if state_action_track[data[0], data[1]] > 0:
                # Comment if no first visit MC
                # state_action_track[data[0]][data[1]] = 0
                # else:
                prob_raw = self.theta_table[data[0]]
                prob = softmax(prob_raw)

                # Define entropy regularizer weight
                entropy_const = 0
                for i in range(self.num_actions):
                    pi_a_s = prob[i]
                    entropy_reg = (
                        entropy_const
                        * (1 / self.num_actions - pi_a_s)
                        / self.num_states
                    )

                    # Update the the tabular parameter theta
                    if i == data[1]:
                        # selected actions
                        self.theta_table[data[0]][data[1]] += alpha_k * (
                            self.lr * G[epi] * (1 - pi_a_s) + entropy_reg
                        )
                    else:
                        self.theta_table[data[0]][i] -= alpha_k * (
                            self.lr * G[epi] * (pi_a_s) + entropy_reg
                        )
