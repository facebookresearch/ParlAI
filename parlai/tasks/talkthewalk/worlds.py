#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.worlds import ExecutableWorld
import json
import os
import copy
from random import choice, randint

BOUNDARIES = {
    'hellskitchen': [3, 3],
    'williamsburg': [2, 8],
    'uppereast': [3, 3],
    'fidi': [2, 3],
    'eastvillage': [3, 4],
}


def is_action(msg, forward=False):
    if forward:
        return msg and (msg == 'ACTION:FORWARD' or msg == 'ACTION : FORWARD')
    return msg and msg.startswith('ACTION')


class Simulator:

    boundaries = None
    neighborhood = None
    agent_location = None
    target_location = None
    landmarks = None

    def __init__(self, opt):
        self.map = Map(opt['ttw_data'])
        self.feature_loader = GoldstandardFeatures(self.map)

    def _get_random_location(self):
        return [
            randint(self.boundaries[0], self.boundaries[2]),
            randint(self.boundaries[1], self.boundaries[3]),
            randint(0, 3),
        ]

    def random_init(self):
        self.neighborhood = choice(list(BOUNDARIES.keys()))
        self.boundaries = [
            randint(0, BOUNDARIES[self.neighborhood][x]) * 2 for x in range(2)
        ]
        self.boundaries = self.boundaries + [x + 3 for x in self.boundaries]
        self.agent_location = self._get_random_location()
        self.target_location = self._get_random_location()

    def init_sim(
        self,
        neighborhood=None,
        boundaries=None,
        start_location=None,
        target_location=None,
    ):

        if not neighborhood:
            self.random_init()
        else:
            self.boundaries = boundaries
            self.neighborhood = neighborhood
            self.agent_location = start_location
            self.target_location = target_location

        self.landmarks, self.target_location = self.map.get_landmarks(
            self.neighborhood, self.boundaries, self.target_location
        )

    def get_agent_location(self):
        return str(
            (self.agent_location[0] - self.boundaries[0]) * 4
            + (self.agent_location[1] - self.boundaries[1])
        )

    def get_text_map(self):
        L = [y for x in zip(*self.landmarks) for y in x]
        txt = '\n'.join([(str(i) + ':' + ' and '.join(x)) for i, x in enumerate(L)])
        txt += (
            '\n'
            + str(self.target_location[0] * 4 + self.target_location[1])
            + ':Target'
            + '\n'
        )
        return txt

    def get_current_view(self):
        return "\n".join(
            [
                'see:' + x
                for x in self.feature_loader.get(self.neighborhood, self.agent_location)
            ]
        )

    def add_view_to_text(self, obs, action=None):
        action = action or obs.get('text')
        if action and is_action(action, forward=True):
            obs['text'] = obs.get('text', '') + self.get_current_view() + '\n'

    def execute_and_write(self, obs, action):
        self.execute(action)
        self.add_view_to_text(obs, action)

    def execute(self, text):
        """
        move the tourist.
        """
        if not is_action(text):
            return

        self.agent_location = self.map.step_aware(
            text, self.agent_location, self.boundaries
        )


class SimulateWorld(ExecutableWorld):

    boundaries = None
    neighborhood = None
    agent_location = None
    target_location = None
    guide = None
    tourist = None

    def __init__(self, opt, agents=None, shared=None):
        super().__init__(opt, agents, shared)

        if agents:
            self.tourist = agents[0]
            self.guide = agents[1]

        if shared:
            self.sim = shared['sim']
        else:
            self.sim = Simulator(opt)
            self.sim.init_sim()

        if agents:
            self.send_map(self.guide)
            self.send_view(self.tourist)

    def send_map(self, agent):
        agent.observe({'text': self.sim.get_text_map()})

    def send_view(self, agent):
        agent.observe({'text': self.sim.get_current_view()})

    def share(self):
        shared = super().share()
        shared['sim'] = self.sim
        return shared

    def execute(self, agent, act):
        self.sim.execute(act['text'])

    def episode_done(self):
        return self.guide_act.startswith('EVALUATE')

    def parley(self):
        act = self.tourist.act()
        while is_action(act['text']):
            self.execute(self.tourist, act)
            obs = self.observe(self.tourist, act)
            if obs is not None:
                self.tourist.observe(obs)
            act = self.tourist.act()
        self.guide.observe(act)
        self.guide_act = self.guide.act()
        obs = self.observe(self.guide, act)

        if obs is not None:
            self.tourist.observe(obs)
        self.update_counters()

    def observe(self, agent, act):
        self.sim.add_view_to_text(act)
        return act


class Map(object):
    """
    Map with landmarks.
    """

    def __init__(self, data_dir, include_empty_corners=True):
        super().__init__()
        self.coord_to_landmarks = dict()
        self.include_empty_corners = include_empty_corners
        self.data_dir = data_dir
        self.landmarks = dict()

        for neighborhood in BOUNDARIES.keys():
            self.coord_to_landmarks[neighborhood] = [
                [[] for _ in range(BOUNDARIES[neighborhood][1] * 2 + 4)]
                for _ in range(BOUNDARIES[neighborhood][0] * 2 + 4)
            ]
            self.landmarks[neighborhood] = json.load(
                open(os.path.join(data_dir, neighborhood, "map.json"))
            )
            for landmark in self.landmarks[neighborhood]:
                coord = self.transform_map_coordinates(landmark)
                self.coord_to_landmarks[neighborhood][coord[0]][coord[1]].append(
                    landmark['type']
                )

    def transform_map_coordinates(self, landmark):
        x_offset = {"NW": 0, "SW": 0, "NE": 1, "SE": 1}
        y_offset = {"NW": 1, "SW": 0, "NE": 1, "SE": 0}

        coord = (
            landmark['x'] * 2 + x_offset[landmark['orientation']],
            landmark['y'] * 2 + y_offset[landmark['orientation']],
        )
        return coord

    def get(self, neighborhood, x, y):
        landmarks = self.coord_to_landmarks[neighborhood][x][y]
        if self.include_empty_corners and len(landmarks) == 0:
            return ['Empty']
        return landmarks

    def get_landmarks(self, neighborhood, boundaries, target_loc):
        landmarks = [[[] for _ in range(4)] for _ in range(4)]
        label_index = (target_loc[0] - boundaries[0], target_loc[1] - boundaries[1])
        for x in range(4):
            for y in range(4):
                landmarks[x][y] = self.get(
                    neighborhood, boundaries[0] + x, boundaries[1] + y
                )

        assert 0 <= label_index[0] < 4
        assert 0 <= label_index[1] < 4

        return landmarks, label_index

    def get_unprocessed_landmarks(self, neighborhood, boundaries):
        landmark_list = []
        for landmark in self.landmarks[neighborhood]:
            coord = self.transform_map_coordinates(landmark)
            if (
                boundaries[0] <= coord[0] <= boundaries[2]
                and boundaries[1] <= coord[1] <= boundaries[3]
            ):
                landmark_list.append(landmark)
        return landmark_list

    def step_aware(self, action, loc, boundaries):
        orientations = ['N', 'E', 'S', 'W']
        steps = dict()
        steps['N'] = [0, 1]
        steps['E'] = [1, 0]
        steps['S'] = [0, -1]
        steps['W'] = [-1, 0]

        new_loc = copy.deepcopy(loc)
        if action == 'ACTION:TURNLEFT':
            # turn left
            new_loc[2] = (new_loc[2] - 1) % 4

        if action == 'ACTION:TURNRIGHT':
            # turn right
            new_loc[2] = (new_loc[2] + 1) % 4

        if action == 'ACTION:FORWARD':
            # move forward
            orientation = orientations[loc[2]]
            new_loc[0] = new_loc[0] + steps[orientation][0]
            new_loc[1] = new_loc[1] + steps[orientation][1]

            new_loc[0] = min(max(new_loc[0], boundaries[0]), boundaries[2])
            new_loc[1] = min(max(new_loc[1], boundaries[1]), boundaries[3])
        return new_loc


class GoldstandardFeatures:
    def __init__(self, map, orientation_aware=False):
        self.map = map
        self.allowed_orientations = {
            'NW': [3, 0],
            'SW': [2, 3],
            'NE': [0, 1],
            'SE': [1, 2],
        }
        self.mod2orientation = {(0, 0): 'SW', (1, 0): 'SE', (0, 1): 'NW', (1, 1): 'NE'}
        self.orientation_aware = orientation_aware

    def get(self, neighborhood, loc):
        if self.orientation_aware:
            mod = (loc[0] % 2, loc[1] % 2)
            orientation = self.mod2orientation[mod]
            if loc[2] in self.allowed_orientations[orientation]:
                return self.map.get(neighborhood, loc[0], loc[1])
            else:
                return ['Empty']
        else:
            return self.map.get(neighborhood, loc[0], loc[1])
