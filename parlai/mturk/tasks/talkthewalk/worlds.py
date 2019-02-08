#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import random
import time
import copy
import numpy as np
import pickle

from joblib import Parallel, delayed

from parlai.core.worlds import MultiAgentDialogWorld
from parlai.mturk.core.agents import MTURK_DISCONNECT_MESSAGE
from parlai.mturk.core.worlds import MTurkOnboardWorld


def _agent_shutdown(agent, timeout):
    agent.shutdown(timeout=timeout)


class TalkTheWalkWorld(MultiAgentDialogWorld):
    """A world where messages from agents can be interpreted as _actions_ in the
    world which result in changes in the environment (are executed). Hence a
    grounded simulation can be implemented rather than just dialogue between
    agents.
    """
    def __init__(self, opt, agents=None, shared=None, world_tag='[NONE]'):
        super().__init__(opt, agents, shared)
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.world_tag = world_tag
        self.replay = opt.get('replay', False)
        self.real_time = opt.get('real_time', False)
        self.replay_bot = opt.get('replay_bot', False)
        self.bot_type = opt.get('bot_type', 'discrete')
        self.logs_file = opt.get('replay_log_file')

        self.actions = ["ACTION:TURNLEFT",
                        "ACTION:TURNRIGHT",
                        "ACTION:FORWARD"]
        self.start_location = None
        self.location = None
        self.target_location = None
        self.orientations = ['N', 'E', 'S', 'W']

        self.neighborhoods = ['hellskitchen',
                              'williamsburg',
                              'fidi',
                              'eastvillage']
        self.boundaries = {}
        self.boundaries['hellskitchen'] = [3, 3]
        self.boundaries['williamsburg'] = [2, 8]
        self.boundaries['eastvillage'] = [3, 4]
        self.boundaries['fidi'] = [2, 3]

        self.steps = {}
        self.steps['N'] = [0, 1]
        self.steps['E'] = [1, 0]
        self.steps['S'] = [0, -1]
        self.steps['W'] = [-1, 0]

        self.min_x, self.min_y, self.max_x, self.max_y = None, None, None, None
        self.landmarks = []
        self.neighborhood = None
        self.start_time = time.time()
        self.total_time = None
        self.num_evaluations = 0

        self.round = 0
        self.status = None
        self.episodeDone = False
        self.acts = []
        if self.replay:
            self.load_data()
            self.load_world(opt['world_idx'])
            self.start_idx = opt.get('start_idx')
        else:
            self.init_world()

    def load_data(self):
        """Load the data for replaying a dialog"""
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 self.logs_file)
        with open(data_path) as f:
            self.data = json.load(f)

    def load_world(self, world_idx):
        """Loads a world into the task when replaying data"""
        if world_idx == -1:
            success_worlds = []
            best_world_len = 1000
            best_world_idx = 1000
            for world_idx in range(len(self.data)):
                if not self.replay_bot:
                    if 40 < len(self.data[world_idx]['dialog']) < 120:
                        break
                else:
                    world = copy.deepcopy(self.data[world_idx])
                    is_success, length = self.is_world_success(world)
                    if is_success:
                        success_worlds.append((world_idx, length))
                        if len(world['dialog']) < best_world_len:
                            best_world_idx = world_idx
                            best_world_len = length
            print(success_worlds)
            world_idx = best_world_idx
        print(world_idx, len(self.data[world_idx]['dialog']))
        self.world_idx = world_idx
        world = self.data[world_idx]
        self.loaded_world = world
        self.neighborhood = world['neighborhood']
        self.target_location = world['target_location']
        self.start_location = world['start_location']
        self.location = self.start_location
        self.landmarks = world['landmarks']
        self.replay_acts = world['dialog']
        self.boundaries = world['boundaries']
        self.min_x, self.min_y, self.max_x, self.max_y = self.boundaries
        self.send_location(self.agents[0])
        self.send_map(self.agents[1])

    def init_world(self):
        """Initializes a new world for the dialog"""
        # first sample neighborhood
        neighborhood_ind = random.randint(0, len(self.neighborhoods) - 1)
        self.neighborhood = self.neighborhoods[neighborhood_ind]

        # Sample 2x2 grid in neighborhood
        self.min_x = random.randint(0, self.boundaries[self.neighborhood][0]) * 2
        self.min_y = random.randint(0, self.boundaries[self.neighborhood][1]) * 2
        self.max_x = self.min_x + 3
        self.max_y = self.min_y + 3

        self.location = [random.randint(self.min_x, self.max_x),
                         random.randint(self.min_y, self.max_y),
                         random.randint(0, 3)]  # x, y, orientation idx
        self.target_location = [random.randint(self.min_x, self.max_x),
                                random.randint(self.min_y, self.max_y),
                                random.randint(0, 3)]  # x, y, orientation idx

        self.start_location = [self.location[0],
                               self.location[1],
                               self.location[2]]

        map_f = os.path.join(self.dir, '{}_map.json'.format(self.neighborhood))
        with open(map_f) as f:
            data = json.load(f)
            for landmark in data:
                if (
                    landmark['x'] * 2 >= self.min_x and
                    landmark['x'] * 2 <= self.max_x and
                    landmark['y'] * 2 >= self.min_y and
                    landmark['y'] * 2 <= self.max_y
                ):
                    self.landmarks.append(landmark)

        self.send_location(self.agents[0])
        self.send_map(self.agents[1])

    def update_location(self, act):
        """Updates the tourist's location based on an action"""
        if act == "ACTION:TURNLEFT":
            self.location[2] = (self.location[2] - 1) % 4
        if act == "ACTION:TURNRIGHT":
            self.location[2] = (self.location[2] + 1) % 4
        if act == "ACTION:FORWARD":
            orientation = self.orientations[self.location[2]]
            self.location[0] += self.steps[orientation][0]
            self.location[1] += self.steps[orientation][1]

            self.location[0] = max(min(self.location[0], self.max_x),
                                   self.min_x)
            self.location[1] = max(min(self.location[1], self.max_y),
                                   self.min_y)

    def send_location(self, agent):
        """Sends the current location to the given agent"""
        msg = {'id': "WORLD_LOCATION",
               'text': {'location': self.location,
                        'boundaries': [self.min_x,
                                       self.min_y,
                                       self.max_x,
                                       self.max_y],
                        'neighborhood': self.neighborhood}}
        agent.observe(msg)

    def send_map(self, agent):
        """Sends the world map to the given agent"""
        msg = {'id': "WORLD_MAP",
               'text': {'landmarks': self.landmarks,
                        'target': self.target_location,
                        'boundaries': [self.min_x,
                                       self.min_y,
                                       self.max_x,
                                       self.max_y]}}
        agent.observe(msg)

    def is_action(self, msg):
        """Returns whether a message is an action from the Tourist"""
        return msg in self.actions

    def episode_done(self):
        return self.episodeDone

    def timeout(self, agent):
        self.status = 'timeout'
        self.causal_agent_id = agent.id
        msg = {'id': "WORLD_TIMEOUT",
               'text': ''}
        agent.observe(msg)

        for other_agent in self.agents:
            if other_agent.id != agent.id:
                msg = {'id': 'WORLD_PARTNER_TIMEOUT',
                       'text': ''}
                other_agent.observe(msg)

    def is_world_success(self, world):
        """Determines whether a given world/dialog yielded a successful
           run of the task. Used when loading a world from data for replay.
        """
        target_location = world['target_location']
        start_location = world['start_location']
        location = start_location
        replay_acts = world['dialog']
        min_x, min_y, max_x, max_y = world['boundaries']
        num_evaluations = 0
        last_grid = None

        def evaluate_location(num_evals, location, target):
            if num_evals == 3:
                return num_evals, False, True
            num_evals += 1
            return (num_evals,
                    (location[0] == target[0] and location[1] == target[1]),
                    False)

        def update_location(act, loc, mi_x, ma_x, mi_y, ma_y):
            if act == "ACTION:TURNLEFT":
                loc[2] = (loc[2] - 1) % 4
            if act == "ACTION:TURNRIGHT":
                loc[2] = (loc[2] + 1) % 4
            if act == "ACTION:FORWARD":
                orientation = self.orientations[loc[2]]
                loc[0] += self.steps[orientation][0]
                loc[1] += self.steps[orientation][1]

                loc[0] = max(min(loc[0], ma_x), mi_x)
                loc[1] = max(min(loc[1], ma_y), mi_y)
            return loc

        for kk, act in enumerate(replay_acts):
            if self.is_action(act['text']):
                location = update_location(act['text'],
                                           location,
                                           min_x,
                                           max_x,
                                           min_y,
                                           max_y)
            elif act['text'] == 'EVALUATE_LOCATION':
                num_evals, done, too_many = evaluate_location(num_evaluations,
                                                              location,
                                                              target_location)
                if done:
                    max_prob = 0
                    max_i_j = None
                    for i in range(len(last_grid)):
                        for j in range(len(last_grid[i])):
                            if last_grid[i][j] > max_prob:
                                max_i_j = (i, j)
                                max_prob = last_grid[i][j]
                    if max_i_j != (location[0] - min_x, location[1] - min_y):
                        return False, -1
                    high_prob = any(any(k >= 0.50 for k in j)
                                    for j in last_grid)
                    max_prob = max(max(j) for j in last_grid)
                    return (True and high_prob, kk)
                elif too_many:
                    return False, -1
            elif act['id'] == 'Guide':
                last_grid = act['text']

        return False, -1

    def replay_actions(self):
        """Replays a loaded dialog in the mturk interface"""
        tourist = self.agents[0]
        guide = self.agents[1]
        cur_time = None
        actions = []
        start = self.start_idx
        time.sleep(5)
        for i in range(start, len(self.replay_acts)):
            act = self.replay_acts[i]
            if self.real_time:
                if cur_time is None:
                    cur_time = act['time']
                else:
                    elapsed = act['time'] - cur_time
                    if not self.is_action(elapsed):
                        elapsed *= 0.75
                        if not self.real_time:
                            elapsed = min(elapsed, 2)
                    time.sleep(elapsed)
                    cur_time = act['time']
            else:
                time.sleep(2)
            if self.is_action(act['text']):
                self.update_location(act['text'])
                act['id'] = 'ACTION'
                tourist.observe(act)
                act['id'] = 'Tourist'
                actions.append(act)
                continue
            if act['text'] == 'EVALUATE_LOCATION':
                done = self.evaluate_location()
                if done:
                    self.episodeDone = True
                    return
            else:
                if self.replay_bot:
                    if act['id'] == 'Tourist' and self.bot_type != 'natural':
                        text = act['text']
                        act['text'] = text[:16]
                    elif act['id'] == 'Guide':
                        grid = act['text']
                        old_grid = np.array(grid)
                        sizes = [9, 19, 39]
                        for i in sizes:
                            new_grid = self.construct_expanded_array(old_grid, i)
                            old_grid = new_grid
                        act['attn_grid'] = new_grid[:37, :37].tolist()
                        act['attn_grid_size'] = sizes[-1] - 2
                        binary_grid = ''
                        mean = np.mean(np.array(grid))
                        for i in range(len(grid)):
                            for j in range(len(grid[i])):
                                num = int(grid[i][j] > mean)
                                binary_grid += str(num)
                        act['show_grid'] = True
                        act['text'] = binary_grid
                guide.observe(act)
                if 'attn_grid' in act:
                    act['attn_grid'] == []
                tourist.observe(act)
        self.episodeDone = True

    def construct_expanded_array(self, grid, size):
        """Constructing a larger attention grid when replaying actions.
           Used when displaying the heat map for the Guide.
        """
        new_grid = np.full((size, size), -1.0)
        new_grid = self.fill_initial(new_grid, grid, size)
        new_grid = self.fill_neighbors(new_grid, size)
        return new_grid

    def neighbor_coords(self, cell, max_size):
        x, y = cell
        X = Y = max_size
        return [(x2, y2) for x2 in range(x - 1, x + 2)
                for y2 in range(y - 1, y + 2)
                if (-1 < x < X and
                -1 < y < Y and
                (x != x2 or y != y2) and
                (0 <= x2 < X) and
                (0 <= y2 < Y))]

    def fill_initial(self, new_g, old_g, size):
        for i in (0, size - 1):
            for j in (range(size)):
                new_g[i, j] = 0
        for j in (0, size - 1):
            for i in range(size):
                new_g[i, j] = 0
        for i in range(1, size, 2):
            for j in range(1, size, 2):
                new_g[i, j] = old_g[(i - 1) // 2, (j - 1) // 2]
        for i in range(1, size - 1, 2):
            for j in range(2, size - 1, 2):
                new_g[i, j] = (new_g[i, j - 1] + new_g[i, j + 1]) / 2
        for i in range(2, size - 1, 2):
            for j in range(1, size - 1, 2):
                new_g[i, j] = (new_g[i - 1, j] + new_g[i + 1, j]) / 2
        return new_g

    def fill_neighbors(self, grid, size):
        for i in range(size):
            for j in range(size):
                if grid[i, j] == -1:
                    neighbors = self.neighbor_coords((i, j), size)
                    neighbor_sum = sum((grid[k, l] for k, l in neighbors))
                    grid[i, j] = neighbor_sum / len(neighbors)
        return grid

    def parley(self):
        if self.replay:
            self.replay_actions()
            return
        while True:
            # Tourist
            agent = self.agents[0]
            act = agent.act(blocking=False)
            if act:
                act['time'] = time.time()
                if act['text'] == MTURK_DISCONNECT_MESSAGE:
                    # episode_done=true so conversations ends
                    self.status = 'disconnect'
                    self.episodeDone = True
                    break
                if self.is_action(act['text']):
                    self.update_location(act['text'])
                self.acts.append(act)
                self.agents[1].observe(act)

            # Guide
            agent = self.agents[1]
            act = agent.act(blocking=False)
            if act:
                act['time'] = time.time()
                self.acts.append(act)
                if act['text'] == MTURK_DISCONNECT_MESSAGE:
                    # episode_done=true so conversations ends
                    self.status = 'disconnect'
                    self.episodeDone = True
                    break
                if act['text'] == 'EVALUATE_LOCATION':
                    done = self.evaluate_location()
                    if done:
                        self.episodeDone = True
                        break
                else:
                    self.agents[0].observe(act)

            time.sleep(0.1)

    def evaluate_location(self):
        self.num_evaluations += 1
        success = (self.location[0] == self.target_location[0] and
                   self.location[1] == self.target_location[1])
        if success:
            print("SUCCESS!!")
            self.status = 'success'
            msg = {'id': 'WORLD_SUCCESS',
                   'text': ''}
            for agent in self.agents:
                agent.observe(msg)
            return True
        else:
            self.status = 'failed'
            if self.num_evaluations < 3:
                msg = {
                    'id': 'Noah',
                    'text': 'Unfortunately, the Tourist is not at the '
                            'target location. You have {} attempt(s) left, '
                            'and you\'ll now receive a bonus of {}c upon '
                            'completion.'.format(
                                str(3 - self.num_evaluations),
                                str(40 - self.num_evaluations * 15)
                            ),
                }
                for agent in self.agents:
                    agent.observe(msg)
                return False
            else:
                msg = {'id': 'WORLD_FAIL',
                       'text': ''}
                for agent in self.agents:
                    agent.observe(msg)
                return True

    def shutdown(self):
        self.total_time = time.time() - self.start_time
        Parallel(
            n_jobs=len(self.agents), backend='threading'
        )(delayed(_agent_shutdown)(agent, timeout=90) for agent in self.agents)

    def review_work(self):
        for agent in self.agents:
            # Disonnects/timeouts are ignored because they never submit the HIT
            agent.approve_work()
            if self.status == 'success':
                if self.num_evaluations == 1:
                    agent.pay_bonus(0.40)
                elif self.num_evaluations == 2:
                    agent.pay_bonus(0.25)
                elif self.num_evaluations == 3:
                    agent.pay_bonus(0.10)

    def save(self):
        """Saves the state of the world"""
        data_path = self.opt['data_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        filename = os.path.join(
            data_path,
            '{}_{}_{}.pkl'.format(
                time.strftime("%Y%m%d-%H%M%S"),
                np.random.randint(0, 1000),
                self.task_type))
        data = {'neighborhood': self.neighborhood,
                'start_location': self.start_location,
                'target_location': self.target_location,
                'location': self.location,
                'status': self.status,
                'dialog': self.acts,
                'landmarks': self.landmarks,
                'tourist_worker_id': self.agents[0].worker_id,
                'tourist_assignment_id': self.agents[0].assignment_id,
                'guide_worker_id': self.agents[1].worker_id,
                'guide_assignment_id': self.agents[1].assignment_id,
                'boundaries': [self.min_x, self.min_y, self.max_x, self.max_y],
                'total_time': self.total_time,
                'version': 1}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print('{}: Data successfully saved at {}.'.format(self.world_tag,
                                                          filename))


class InstructionWorld(MTurkOnboardWorld):

    def parley(self):
        self.mturk_agent.act()
        self.episodeDone = True
