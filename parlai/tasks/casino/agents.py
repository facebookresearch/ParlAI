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

WELCOME_MESSAGE = "Negotiate with your opponent to decide who gets how many items of each kind. There are three kinds of packages: Food, Water, and Firewood. Each has a quantity of 3. Try hard to get as much value as you can, while still leaving your partner satisfied and with a positive perception about you. If you fail to come to an agreement, both parties get 5 points. Refer to the following preference order and arguments for your negotiation: \n\nFood\nValue: {food_val} points for each package\nArgument: {food_argument}\n\nWater\nValue: {water_val} points for each package\nArgument: {water_argument}\n\nFirewood\nValue: {firewood_val} points for each package\nArgument: {firewood_argument}\n"


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


def get_utterance_text(utterance):

    if utterance['text'] == '<DUMMY>':
        return ''

    # the utterance is not a dummy one at this point
    if utterance['text'] != 'Submit-Deal':
        # simply return it
        return utterance['text']

    # if it is a Submit-Deal -> attach task_data
    txt = f"{utterance['text']} What I get- Food:{utterance['task_data']['issue2youget']['Food']}, Water: {utterance['task_data']['issue2youget']['Water']}, Firewood: {utterance['task_data']['issue2youget']['Firewood']}; What you get- Food:{utterance['task_data']['issue2theyget']['Food']}, Water: {utterance['task_data']['issue2theyget']['Water']}, Firewood: {utterance['task_data']['issue2theyget']['Firewood']}"

    return txt


class CasinoTeacher(Teacher):
    """
    A negotiation teacher that loads the CaSiNo data from
    https://github.com/kushalchawla/CaSiNo.

    Each dialogue is converted into two datapoints, one from the perspective of each
    participant.
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
            ] = 'mturk_agent_1'  # id of the agent whose perspective will be used in this dialog
            episodes.append(episode)

            episode = copy.deepcopy(dialogue)
            episode[
                'perspective'
            ] = 'mturk_agent_2'  # id of the agent whose perspective will be used in this dialog
            episodes.append(episode)

        self.episodes = episodes

        # add dummy data to ensure that every chat begins with a teacher utterance (THEM) and ends at the agent's utterance (YOU). This is done for uniformity while parsing the data. It makes the code simpler and easier to read than DealNoDeal counterpart.
        for ix, episode in enumerate(self.episodes):

            chat_logs = episode['chat_logs']
            perspective = episode['perspective']

            if chat_logs[0]['id'] == perspective:
                # chat must start with a teacher; add dummy utterance
                dummy_utterance = {
                    'text': '<DUMMY>',
                    'task_data': {},
                    'id': 'mturk_agent_1'
                    if perspective == 'mturk_agent_2'
                    else 'mturk_agent_2',
                }

                chat_logs = [dummy_utterance] + chat_logs

            if chat_logs[-1]['id'] != perspective:
                # chat must end with the agent; add dummy utterance
                dummy_utterance = {
                    'text': '<DUMMY>',
                    'task_data': {},
                    'id': 'mturk_agent_1'
                    if perspective == 'mturk_agent_1'
                    else 'mturk_agent_2',
                }

                chat_logs = chat_logs + [dummy_utterance]

            self.episodes[ix]['chat_logs'] = chat_logs

    def reset(self):
        super().reset()
        self.episode_idx = self.data_offset - self.step_size
        self.dialogue_idx = None
        self.perspective = None
        self.dialogue = None
        self.output = None
        self.expected_response = None
        self.epochDone = False

    def num_examples(self):
        """
        Lets return the the number of responses that an agent would generate in one
        epoch + 1 count for every output.

        This will include special utterances for submit-deal, accept-deal, and reject-
        deal.
        """
        num_exs = 0

        for episode in self.episodes:

            for utt in episode['chat_logs']:
                if utt['text'] != '<DUMMY>':
                    # skip the dummy utterances
                    num_exs += 1

        return (num_exs // 2) + len(
            self.episodes
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
        if self.expected_response is not None:
            self.metrics.evaluate_response(observation, self.expected_response)
            self.expected_response = None
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
        """
        Starting a dialogue should be the same as continuing a dialogue but with just
        one difference: it will attach the welcome note to the teacher's utterance.

        Each dialogue has two agents possible: mturk_agent_1 or mturk_agent_2. One of
        them will act as the perspective for this episode.
        """

        episode = self.episodes[self.episode_idx]
        self.perspective = episode['perspective']
        self.other_id = (
            'mturk_agent_1' if self.perspective == 'mturk_agent_2' else 'mturk_agent_2'
        )

        part_info = episode['participant_info'][self.perspective]
        part_info_other = episode['participant_info'][self.other_id]

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
        self.output = {
            'your_points_scored': part_info['outcomes']['points_scored'],
            'how_satisfied_is_your_partner': part_info_other['outcomes'][
                'satisfaction'
            ],
            'how_much_does_your_partner_like_you': part_info_other['outcomes'][
                'opponent_likeness'
            ],
        }

        self.dialogue_idx = -1

        action = self._continue_dialogue()
        if action['text']:
            # This is non-empty; meaning the teacher starts the conversation and has something to say.
            action['text'] = f"{welcome}\n{action['text']}"
        else:
            # text is empty, meaning that the teacher did not start the conversation but the empty string is just a result of the dummy teacher utterance added in _setup_data
            action['text'] = welcome

        action['meta-info'] = welcome_values

        return action

    def _continue_dialogue(self):
        """
        Return an action object.

        From the perspective of a specific agent's id, all utterances authored by the
        other agent are coming from the teacher as the text of the action object, and
        all utterances authored by this agent appear as the labels.
        """
        action = {}
        # Fill in teacher's message (THEM)
        self.dialogue_idx += 1
        if self.dialogue_idx < len(self.dialogue):
            # this is a usual dialogue teacher-agent pair; return the teacher's utterance as action text.
            utterance = self.dialogue[self.dialogue_idx]
            assert utterance['id'] != self.perspective
            utterance_text = get_utterance_text(
                utterance
            )  # will take care of special submit-deal utterance and dummy utterances
            action['text'] = utterance_text

            if action['text'] == 'Reject-Deal':
                # merge with the next dialogue_idx since that is from the same participant while this code assumes alternative utterances.
                self.dialogue_idx += 1  # we know that this will be valid
                utterance = self.dialogue[self.dialogue_idx]
                assert utterance['id'] != self.perspective
                utterance_text = get_utterance_text(
                    utterance
                )  # will take care of special submit-deal utterance and dummy utterances
                action['text'] = action['text'] + ' ' + utterance_text
        else:
            # the primary dialogue is over; now is the time to return the output of this dialogue
            action[
                'text'
            ] = f"Your points scored: {self.output['your_points_scored']}, How satisfied is your partner: {self.output['how_satisfied_is_your_partner']}, How much does your partner like you: {self.output['how_much_does_your_partner_like_you']}"

        # Fill in learner's response (YOU)
        self.dialogue_idx += 1
        self.expected_response = None
        if self.dialogue_idx < len(self.dialogue):
            # usual dialogue going on; return the agent's utterance as the labels
            utterance = self.dialogue[self.dialogue_idx]
            assert (
                utterance['id'] == self.perspective
            ), f"id: {utterance['id']}, perspect: {self.perspective}"
            utterance_text1 = get_utterance_text(
                utterance
            )  # will take care of special submit-deal utterance and dummy utterances

            utterance_text2 = ''
            if utterance_text1 == 'Reject-Deal':
                # merge with the next dialogue_idx since that is from the same participant while this code assumes alternative utterances.
                self.dialogue_idx += 1  # we know that this will be valid
                utterance = self.dialogue[self.dialogue_idx]
                assert utterance['id'] == self.perspective
                utterance_text2 = get_utterance_text(
                    utterance
                )  # will take care of special submit-deal utterance and dummy utterances

            self.expected_response = (
                [utterance_text1 + ' ' + utterance_text2]
                if (utterance_text1 + ' ' + utterance_text2).strip()
                else None
            )
        else:
            # no label required when the primary dialogue is complete
            pass

        if self.expected_response:
            # since labels is automatically renamed to eval_labels for valid/test, doing just this takes care of everything. Ensures that labels can atleast be accessed regardless of the datatype.
            action['labels'] = self.expected_response

        if self.dialogue_idx >= len(self.dialogue):
            self.dialogue_idx = None
            action['episode_done'] = True
        else:
            action['episode_done'] = False

        return action


class DefaultTeacher(CasinoTeacher):
    pass
