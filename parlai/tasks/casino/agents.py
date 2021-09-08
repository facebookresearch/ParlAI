#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import Teacher
from .build import build
from parlai.utils.io import PathManager
import json
import os
import random
import copy

WELCOME_MESSAGE = "Negotiate with your opponent to decide who gets how many of each item. There are three quantities each of Food, Water, and Firewood. Try hard to get as much value as you can, while still leaving your partner satisfied and with a positive perception about you. If you fail to come to an agreement, both parties get 5 points. Refer to the following preference order and arguments for your negotiation: \n\nFood\nValue:{food_val} points for each package\nArgument:{food_argument}\n\nWater\nValue:{water_val} points for each package\nArgument:{water_argument}\n\nFirewood\nValue:{firewood_val} points for each package\nArgument:{firewood_argument}\n"


def get_welcome_values(part_info):

    value2points = {'High': 5, 'Medium': 4, 'Low': 3}

    issue2points = {v: value2points[k] for k, v in part_info['value2issue'].items()}
    issue2reason = {
        v: part_info['value2reason'][k] for k, v in part_info['value2issue'].items()
    }

    welcome_values = {}
    for issue in ['Food', 'Water', 'Firewood']:
        welcome_values[issue.lower() + '_val'] = issue2points[issue]
        welcome_values[issue.lower() + '_argument'] = issue2reason[issue]

    return welcome_values


class CasinoTeacher(Teacher):
    """
    A negotiation teacher that loads the CaSiNo data from https://github.com/kushalchawla/CaSiNo.

    Each dialogue is converted into two datapoints, one from the perspective of each participant.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.datatype = opt['datatype'].split(':')[0]
        self.datatype_ = opt['datatype']
        self.random = self.datatype_ == 'train'
        build(opt)

        filename = self.datatype
        data_path = os.path.join(
            opt['datapath'], 'casino', 'casino_' + filename + '.json'
        )

        if shared and 'data' in shared:
            self.episodes = shared['episodes']
        else:
            self._setup_data(data_path)
        print(f"Total episodes: {self.num_episodes()}")

        # for ordered data in batch mode (especially, for validation and
        # testing), each teacher in the batch gets a start index and a step
        # size so they all process disparate sets of the data
        self.step_size = opt.get('batchsize', 1)
        self.data_offset = opt.get('batchindex', 0)

        self.reset()

    def _setup_data(self, data_path):
        print('loading: ' + data_path)
        with PathManager.open(data_path) as data_file:

            dialogues = json.load(data_file)
            episodes = []

            for dialogue in dialogues:

                # divide the dialogue into two perspectives, one for each participant
                episode = copy.deepcopy(dialogue)
                episode[
                    'perspective'
                ] = (
                    'mturk_agent_1'
                )  # id of the agent whose perspective will be used in this dialog
                episodes.append(episode)

                episode = copy.deepcopy(dialogue)
                episode[
                    'perspective'
                ] = (
                    'mturk_agent_2'
                )  # id of the agent whose perspective will be used in this dialog
                episodes.append(episode)

            self.episodes = episodes

    def reset(self):
        super().reset()
        self.episode_idx = self.data_offset - self.step_size
        self.dialogue_idx = None
        self.perspective = None
        self.expected_reponse = None
        self.epochDone = False

    def num_examples(self):
        """
        Lets simply return the total number of utterances in all the dialogues. This will include special utterances for submit-deal, accept-deal, and reject-deal.

        Half of this quantity will correspond to the number of responses that an agent would generate in one epoch.
        """
        num_exs = 0

        for episode in self.episodes:
            num_exs += len(episode['chat_logs'])

        return (
            num_exs // 2
        )  # since each dialogue was converted into 2 perspectives, one for each participant: see _setup_data

    def num_episodes(self):
        return len(self.episodes)

    def share(self):
        shared = super().share()
        shared['episodes'] = self.episodes
        return shared

    def observe(self, observation):
        """
        Process observation for metrics.
        """
        if self.expected_reponse is not None:
            self.metrics.evaluate_response(observation, self.expected_reponse)
            self.expected_reponse = None
        return observation

    def act(self):
        if self.dialogue_idx is not None:
            # continue existing conversation
            return self._continue_dialogue()
        elif self.random:
            # if random, then select the next random example
            self.episode_idx = random.randrange(len(self.episodes))
            return self._start_dialogue()
        elif self.episode_idx + self.step_size >= len(self.episodes):
            # end of examples
            self.epochDone = True
            return {'episode_done': True}
        else:
            # get next non-random example
            self.episode_idx = (self.episode_idx + self.step_size) % len(self.episodes)
            return self._start_dialogue()

    def _start_dialogue(self):

        episode = self.episodes[self.episode_idx]
        self.perspective = episode['perspective']

        part_info = episode['participant_info'][self.perspective]
        welcome_values = get_welcome_values(part_info)

        welcome = WELCOME_MESSAGE.format(
            food_val=welcome_values['food_val'],
            water_val=welcome_values['water_val'],
            firewood_val=welcome_values['firewood_val'],
            food_argument=welcome_values['food_argument'],
            water_argument=welcome_values['water_argument'],
            firewood_argument=welcome_values['firewood_argument'],
        )

        self.dialogue = episode['chat_logs']
        self.output = part_info['outcomes']
        # The dialogue should end with accept deal
        assert (
            self.dialogue[-1]['text'] == 'Accept-Deal'
        ), 'The dialogue should end with accept deal.'

        self.dialogue_idx = -1
        if self.dialogue[0]['id'] != self.perspective:
            # the other party (or the teacher) starts the dialogue.
            action = self._continue_dialogue()
            action['text'] = welcome + '\n' + action['text']
        else:
            # the agent (whose perspective we are in) starts the dialogue.
            action = self._continue_dialogue(skip_teacher=True)
            action['text'] = welcome

        return action

    def _continue_dialogue(self, skip_teacher=False):
        action = {}

        # Fill in teacher's message (THEM)
        if not skip_teacher:
            self.dialogue_idx += 1
            if self.dialogue_idx >= len(self.dialogue):
                action['text'] = SELECTION_TOKEN
            else:
                utterance = self.dialogue[self.dialogue_idx]
                assert utterance['id'] != self.perspective
                action['text'] = utterance['text']

        # Fill in learner's response (YOU)
        self.dialogue_idx += 1
        if self.datatype.startswith('train'):
            if self.dialogue_idx >= len(self.dialogue):
                # the agent should finish by reporting what they think the
                # agreed decision was
                self.expected_reponse = [' '.join(self.output)]
            else:
                sentence = self.dialogue[self.dialogue_idx]
                assert sentence[0] == YOU_TOKEN
                self.expected_reponse = [' '.join(sentence[1:])]
            action['labels'] = self.expected_reponse

        if self.dialogue_idx >= len(self.dialogue):
            self.dialogue_idx = None
            action['episode_done'] = True
        else:
            action['episode_done'] = False

        return action


class DefaultTeacher(CasinoTeacher):
    pass
