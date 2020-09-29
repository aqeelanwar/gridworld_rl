"""
Name: gridworld_rl
Description: Tabular Reinforcement Learning using policy iteration (REINFORCE) on gridworld problem
Author: Aqeel Anwar (aqeel.anwar@gatech.edu)

Maze visualization: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import time
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

# Define colors to be used in maze
GOAL_COLOR = "#D8B25C"
AGENT_COLOR = "#A5AB81"
WALL_COLOR = "#F24F5C"
GRID_COLOR = "#595959"


class Maze(tk.Tk, object):
    def __init__(
        self,
        name,
        state_nature,
        MAZE_H,
        MAZE_W,
        UNIT,
        hell_coord,
        goal_coord,
        origin_coord,
        opposite_goal,
    ):
        super(Maze, self).__init__()
        self.action_space = ["u", "d", "l", "r"]
        self.n_actions = len(self.action_space)
        if state_nature == "GPS":
            self.n_states = MAZE_H * MAZE_W
        elif state_nature == "SONAR":
            self.n_states = 3 ** self.n_actions

        self.title(name)
        self.geometry("{0}x{1}".format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.MAZE_H = MAZE_H
        self.MAZE_W = MAZE_W
        self.UNIT = UNIT
        self.hell_coord = hell_coord
        self.goal_coord = goal_coord
        self.origin_coord = origin_coord
        self.hell_array = []
        self._build_maze()
        self.cell_num_to_state_name = self.generate_cell_num_to_name()
        self.state_name_to_number = self.gen_state_name_to_number()
        self.opposite_goal = opposite_goal
        self.state_nature = state_nature
        # reward_empty_state = 'zero' or 'distance_based'
        self.reward_empty_state = 'zero'

    def _build_maze(self):
        self.canvas = tk.Canvas(
            self,
            bg="white",
            height=self.MAZE_H * self.UNIT,
            width=self.MAZE_W * self.UNIT,
        )

        # create grids
        for c in range(0, self.MAZE_W * self.UNIT, self.UNIT):
            x0, y0, x1, y1 = c, 0, c, self.MAZE_H * self.UNIT
            self.canvas.create_line(x0, y0, x1, y1, fill=GRID_COLOR)
        for r in range(0, self.MAZE_H * self.UNIT, self.UNIT):
            x0, y0, x1, y1 = 0, r, self.MAZE_W * self.UNIT, r
            self.canvas.create_line(x0, y0, x1, y1, fill=GRID_COLOR)

        # create origin
        origin = np.array(
            [
                int(self.origin_coord[0]) * self.UNIT + 20,
                int(self.origin_coord[1]) * self.UNIT + 20,
            ]
        )

        # create walls
        for h_coord in self.hell_coord:
            hell_center = [int(self.UNIT / 2), int(self.UNIT / 2)] + np.array(
                [self.UNIT * h_coord[0], self.UNIT * h_coord[1]]
            )
            self.hell = self.canvas.create_rectangle(
                hell_center[0] - 15,
                hell_center[1] - 15,
                hell_center[0] + 15,
                hell_center[1] + 15,
                fill=WALL_COLOR,
                outline="",
            )
            self.hell_array.append(self.canvas.coords(self.hell))

        # create goal
        oval_center = [int(self.UNIT / 2), int(self.UNIT / 2)] + np.array(
            [self.UNIT * self.goal_coord[0], self.UNIT * self.goal_coord[1]]
        )
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15,
            oval_center[1] - 15,
            oval_center[0] + 15,
            oval_center[1] + 15,
            fill=GOAL_COLOR,
            outline="",
        )

        # create agent rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15,
            origin[1] - 15,
            origin[0] + 15,
            origin[1] + 15,
            fill=AGENT_COLOR,
            outline="",
        )

        # pack all
        self.canvas.pack()

    def cell_num_to_coordinate(self, cell_num):
        x = cell_num % self.MAZE_W
        y = cell_num // self.MAZE_W
        return [x, y]

    def generate_cell_num_to_name(self):
        cell_num_to_name = {}
        for cell_num in range(self.MAZE_W * self.MAZE_H):
            cell_coord = self.cell_num_to_coordinate(cell_num)

            # check what is up
            up_coord = [cell_coord[0], cell_coord[1] - 1]  # corner case top
            if up_coord[1] >= 0:  # i.e you can move upwards
                if up_coord in self.hell_coord:
                    u = "H"
                elif up_coord == self.goal_coord:
                    u = "G"
                else:
                    u = "N"
            else:
                # depends what you want
                u = "H"

            # check what is down
            down_coord = [cell_coord[0], cell_coord[1] + 1]  # corner case top
            if down_coord[1] < self.MAZE_H:  # i.e you can move upwards
                if down_coord in self.hell_coord:
                    d = "H"
                elif down_coord == self.goal_coord:
                    d = "G"
                else:
                    d = "N"
            else:
                # depends what you want
                d = "H"

            # check what is right
            right_coord = [cell_coord[0] + 1, cell_coord[1]]  # corner case top
            if right_coord[0] < self.MAZE_W:  # i.e you can move upwards
                if right_coord in self.hell_coord:
                    r = "H"
                elif right_coord == self.goal_coord:
                    r = "G"
                else:
                    r = "N"
            else:
                # depends what you want
                r = "H"

            # check what is left
            left_coord = [cell_coord[0] - 1, cell_coord[1]]  # corner case top
            if left_coord[0] >= 0:  # i.e you can move upwards
                if left_coord in self.hell_coord:
                    l = "H"
                elif left_coord == self.goal_coord:
                    l = "G"
                else:
                    l = "N"
            else:
                # depends what you want
                l = "H"

            state_name = u + d + r + l
            cell_num_to_name[cell_num] = state_name
        return cell_num_to_name

    def reset(self):
        self.update()
        # time.sleep(0.01)
        self.canvas.delete(self.rect)
        origin = np.array(
            [
                int(self.origin_coord[0]) * self.UNIT + 20,
                int(self.origin_coord[1]) * self.UNIT + 20,
            ]
        )
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15,
            origin[1] - 15,
            origin[0] + 15,
            origin[1] + 15,
            fill=AGENT_COLOR,
        )
        # return observation
        state_coords = self.canvas.coords(self.rect)
        # state = state_coords[1]/origin[0]
        # TODO return starting position state number
        state = self.cell_num_to_state_name[0]
        state = self.state_name_to_number[state]
        return state, self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        out_of_the_box = True
        if action == 0:  # up
            if s[1] > self.UNIT:
                base_action[1] -= self.UNIT
                out_of_the_box = False
        elif action == 1:  # down
            if s[1] < (self.MAZE_H - 1) * self.UNIT:
                base_action[1] += self.UNIT
                out_of_the_box = False
        elif action == 2:  # right
            if s[0] < (self.MAZE_W - 1) * self.UNIT:
                base_action[0] += self.UNIT
                out_of_the_box = False
        elif action == 3:  # left
            if s[0] > self.UNIT:
                base_action[0] -= self.UNIT
                out_of_the_box = False

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state
        state_ = int(self.MAZE_W * (s_[1] // self.UNIT) + s_[0] // self.UNIT)
        state = int(self.MAZE_W * (s[1] // self.UNIT) + s[0] // self.UNIT)

        state_cartesian = [int(state % self.MAZE_H), int(state // self.MAZE_W)]
        state__cartesian = [int(state_ % self.MAZE_H), int(state_ // self.MAZE_W)]

        # Define rewards
        if not self.opposite_goal:
            if s_ == self.canvas.coords(self.oval):
                reward = 1
                done = True
                win = 1
                s_ = "terminal"
            elif s_ in self.hell_array:
                reward = -1
                done = True
                s_ = "terminal"
                win = 0
            else:
                if self.reward_empty_state=='distance_based':
                    d_state = sum(
                        abs(np.array(self.goal_coord) - np.array(state_cartesian))
                    )
                    d_state_ = sum(
                        abs(np.array(self.goal_coord) - np.array(state__cartesian))
                    )
                    if d_state_ < d_state:
                        reward = 0.1
                    else:
                        reward = -0.1

                elif self.reward_empty_state == 'zero':
                    reward = 0



                done = False
                win = 0
            if out_of_the_box:
                done = True
                reward = -1

        # Define rewards
        elif self.opposite_goal:
            if s_ == self.canvas.coords(self.oval):
                reward = -1
                done = True
                win = 0
                s_ = "terminal"
            elif s_ in self.hell_array:
                reward = 1
                done = True
                s_ = "terminal"
                win = 1
            else:
                # reward = -0.01
                reward = 0
                done = False
                win = 0
            if out_of_the_box:
                done = True
                reward = -1

        if self.state_nature == "SONAR":
            state_ = self.cell_num_to_state_name[state_]
            state_ = self.state_name_to_number[state_]
        elif self.state_nature == "GPS":
            state_ = state_

        return state_, s_, reward, done, win

    def render(self):
        # time.sleep(0.01)
        self.update()

    def gen_state_name_to_number(self):
        tate_vec_len = 4
        state_elemets = ["N", "G", "H"]
        state_name_to_number = {}

        iter = 0
        for i in state_elemets:
            for j in state_elemets:
                for k in state_elemets:
                    for l in state_elemets:
                        name = i + j + k + l
                        state_name_to_number[name] = iter
                        iter += 1

        num_states = len(state_name_to_number)
        return state_name_to_number


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == "__main__":
    env = Maze()
    env.after(100, update)
    env.mainloop()
