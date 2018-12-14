from collections import defaultdict
from parlai.core.worlds import DialogPartnerWorld, ExecutableWorld
from parlai.tasks.talkthewalk.ttw.dict import Dictionary, LandmarkDictionary, ActionAgnosticDictionary, ActionAwareDictionary, TextrecogDict, \
    START_TOKEN, END_TOKEN
import json
import os
import copy

BOUNDARIES = {
    'hellskitchen': [3, 3],
    'williamsburg': [2, 8],
    'uppereast':    [3, 3],
    'fidi':         [2, 3],
    'eastvillage':  [3, 4],
}

class SimulateWorld(ExecutableWorld):

    boundaries = None
    neighborhood = None
    location_history = []
    agent_location = None
    target_location = None


    def __init__(self, opt, agents=None, shared=None):
        super().__init__(opt, agents, shared)
        if shared:
            self.map = shared['map']
            self.feature_loader = shared['feature_loader']

        self.map = Map(self.opt['ttw_data'],
                BOUNDARIES.keys(),
                include_empty_corners=True)
        self.feature_loader = GoldstandardFeatures(self.map)

    def share(self):
        shared_data = super().share()
        shared_data['map'] = self.map
        shared_data['feature_loader'] = self.feature_loader
        return shared_data

    def parley(self):
        acts = self.acts
        agents = self.agents
        tourist = agents[0]
        guide = agents[1]

        #allow tourist to move indefinately and then speak
        while True:
            acts[0] = tourist.act()
            acts[0] = tourist.observe(self.observe(tourist, acts[0]))
            if not acts[0].get('text', '').startswith('ACTION'):
                break

        guide.observe(self.observe(guide, acts[0]))
        acts[1] = guide.act()
        tourist.observe(self.observe(tourist, acts[1]))
        self.update_counters()

    def episode_done(self):
        return self.acts[1].startswith('EVALUATE')

    def observe(self, agent, act):
        act = defaultdict(list, copy.copy(act))
        self.boundaries = act.get('boundaries') or self.boundaries
        self.neighborhood = act.get('neighborhood') or self.neighborhood
        self.agent_location = act.get('start_location') or \
            self.agent_location
        self.target_location = copy.copy(act.get('target_location')) or \
            self.target_location
        act.pop('start_location', None)
        act.pop('target_location', None)

        text = act.get('text')

        if text and text.startswith('ACTION'):
            self.agent_location = self.map.step_aware(
                    text,
                    self.agent_location,
                    self.boundaries)

            if text == ('ACTION:FORWARD'):
                act['see'] = self.feature_loader.get(
                        self.neighborhood,
                        self.agent_location)
                act['text']+='\n'+" ".join(act['see'])

            #tourist agent only uses location to calculate deltas
            act['location'] = self.agent_location

        # if agent.id == 'guide':
        #     act['landmarks'], act['target_location'] = \
        #         self.map.get_landmarks(self.neighborhood, self.boundaries,
        #                 self.target_location)

        print('act', act);
        return act


class Map(object):
    """Map with landmarks"""

    def __init__(self, data_dir, neighborhoods, include_empty_corners=True):
        super(Map, self).__init__()
        self.coord_to_landmarks = dict()
        self.include_empty_corners = include_empty_corners
        self.landmark_dict = LandmarkDictionary()
        self.data_dir = data_dir
        self.landmarks = dict()

        for neighborhood in neighborhoods:
            self.coord_to_landmarks[neighborhood] = [[[] for _ in range(BOUNDARIES[neighborhood][1] * 2 + 4)]
                                                     for _ in range(BOUNDARIES[neighborhood][0] * 2 + 4)]
            self.landmarks[neighborhood] = json.load(open(os.path.join(data_dir, neighborhood, "map.json")))
            for landmark in self.landmarks[neighborhood]:
                coord = self.transform_map_coordinates(landmark)
                # landmark_idx = self.landmark_dict.encode(landmark['type'])
                self.coord_to_landmarks[neighborhood][coord[0]][coord[1]].append(landmark['type'])

    def transform_map_coordinates(self, landmark):
        x_offset = {"NW": 0, "SW": 0, "NE": 1, "SE": 1}
        y_offset = {"NW": 1, "SW": 0, "NE": 1, "SE": 0}

        coord = (landmark['x'] * 2 + x_offset[landmark['orientation']],
                 landmark['y'] * 2 + y_offset[landmark['orientation']])
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
                landmarks[x][y] = self.get(neighborhood, boundaries[0] + x, boundaries[1] + y)

        assert 0 <= label_index[0] < 4
        assert 0 <= label_index[1] < 4

        return landmarks, label_index

    def get_unprocessed_landmarks(self, neighborhood, boundaries):
        landmark_list = []
        for landmark in self.landmarks[neighborhood]:
            coord = self.transform_map_coordinates(landmark)
            if boundaries[0] <= coord[0] <= boundaries[2] and boundaries[1] <= coord[1] <= boundaries[3]:
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
        self.allowed_orientations = {'NW': [3, 0], 'SW': [2, 3], 'NE': [0, 1], 'SE': [1, 2]}
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
