#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from parlai.core.teachers import FixedDialogTeacher
from . import tm_utils
import json


class SelfDialogueTeacher(FixedDialogTeacher):
    """
    Teacher for written two-person dialogues with labels being responses for
    the previous statement. The data is traversed twice (doubled), once for
    modelling USER replies and once for modelling ASSISTANT replies.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        opt['fn'] = "self-dialogs.json"

        if 'exclude-invalid-data' not in opt:
            opt['exclude-invalid-data'] = True

        if shared and 'convos' in shared:
            # another instance was set up already, just reference its data
            self.convos = shared['convos']
            self.ep_cheat_sheet = shared['ep_cheat_sheet']
        else:
            # need to set up data from scratch
            self.ep_cheat_sheet = {}  # Stores imp. info. about each episode
            data_path = tm_utils._path(opt)
            self._setup_data(data_path, opt)

        self.reset()

    def _setup_data(self, data_path, opt):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.convos = json.load(data_file)
        # Pre-processing
        convos_update = []
        for convo in self.convos:
            conversation = convo['utterances']
            # Filter out single greet messages
            if len(conversation) > 1:
                self.ep_cheat_sheet[
                    len(self.ep_cheat_sheet)
                ] = tm_utils.gen_ep_cheatsheet(conversation)
                convos_update += [conversation]
        self.convos = convos_update

    def num_examples(self):
        ctr = 0
        for ep in self.ep_cheat_sheet:
            ctr += (
                self.ep_cheat_sheet[ep][tm_utils.USER_NUM_EX]
                + self.ep_cheat_sheet[ep][tm_utils.ASSIS_NUM_EX]
            )
        return ctr

    def num_episodes(self):
        # For two passes over the data: Once to teach USER and once to teach ASSISTANT
        return len(self.convos) * 2

    def get(self, episode_idx, entry_idx):
        conversation = self.convos[episode_idx % len(self.convos)]
        if episode_idx < len(self.convos):
            # USER then ASSISTANT mode [First pass]
            ep_done = (
                entry_idx * 2 == self.ep_cheat_sheet[episode_idx][tm_utils.LAST_USER]
            )
            predecessor = conversation[entry_idx * 2]['text']
            successor = conversation[entry_idx * 2 + 1]['text']
        else:
            # ASSISTANT then USER mode [Second pass]
            ep_done = (
                entry_idx * 2 + 1
                == self.ep_cheat_sheet[episode_idx % len(self.convos)][
                    tm_utils.LAST_ASSISTANT
                ]
            )
            predecessor = conversation[entry_idx * 2 + 1]['text']
            successor = conversation[entry_idx * 2 + 2]['text']

        action = {
            'id': self.id,
            'text': predecessor,
            'episode_done': ep_done,
            'labels': [successor],
        }

        return action


class WozDialogueTeacher(FixedDialogTeacher):
    """
    Teacher for spoken two-person dialogues with labels being responses for
    the previous statement. The data is traversed twice (doubled), once for
    modelling USER replies and once for modelling ASSISTANT replies.
    """

    def __init__(self, opt, shared=None):
        opt['fn'] = "woz-dialogs.json"
        super().__init__(opt)

        if 'exclude-invalid-data' not in opt:
            opt['exclude-invalid-data'] = True

        if shared and 'convos' in shared:
            # another instance was set up already, just reference its data
            self.convos = shared['convos']
            self.episode_map = shared['episode_map']
            self.ep_cheat_sheet = shared['ep_cheat_sheet']
            self.num_ex = shared['num_ex']
        else:
            # need to set up data from scratch
            self.ep_cheat_sheet = {}  # Stores imp. info. about each episode

            # Not all episodes have relevant examples for both USER and ASSISTANT
            # episode_map keeps track of which episode index is useful for which speaker
            # Need to do this otherwise might end up with a situation where we cannot return anything in action
            self.episode_map = {}
            self.episode_map["U"] = {}
            self.episode_map["A"] = {}
            self.num_ex = 0
            data_path = tm_utils._path(opt)
            self._setup_data(data_path, opt)

        self.reset()

    def _setup_data(self, data_path, opt):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.convos = json.load(data_file)
        # Pre-processing
        convos_update = []
        for convo in self.convos:
            conversation, corrupted = tm_utils.smoothen_convo(convo, opt)
            # Filter out single greet messages and corrupted examples
            if len(conversation) > 1 and not corrupted:
                actual_ep_idx = len(self.ep_cheat_sheet)
                self.ep_cheat_sheet[actual_ep_idx] = tm_utils.gen_ep_cheatsheet(
                    conversation
                )
                curr_cheatsheet = self.ep_cheat_sheet[len(self.ep_cheat_sheet) - 1]
                # calc number of examples (done here to prevent double counting if done later)
                self.num_ex += (
                    curr_cheatsheet[tm_utils.USER_NUM_EX]
                    + curr_cheatsheet[tm_utils.ASSIS_NUM_EX]
                )
                # User example exists
                if curr_cheatsheet[tm_utils.USER_NUM_EX] != 0:
                    u_idx = len(self.episode_map["U"])
                    self.episode_map["U"][u_idx] = actual_ep_idx
                # Assistant example exists
                if curr_cheatsheet[tm_utils.ASSIS_NUM_EX] != 0:
                    a_idx = len(self.episode_map["A"])
                    self.episode_map["A"][a_idx] = actual_ep_idx
                convos_update += [conversation]
        self.convos = convos_update

    def num_examples(self):
        return self.num_ex

    def num_episodes(self):
        # For two passes over the data: Once to teach USER and once to teach ASSISTANT
        return len(self.episode_map["U"]) + len(self.episode_map["A"])

    def get(self, episode_idx, entry_idx):
        starts_at_odd = False
        if episode_idx < len(self.episode_map["U"]):
            # USER then ASSISTANT mode [First pass]
            true_idx = self.episode_map["U"][episode_idx]
            conversation = self.convos[true_idx]
            convo_cheat_sheet = self.ep_cheat_sheet[true_idx]
            first_entry, last_entry = (
                convo_cheat_sheet[tm_utils.FIRST_USER],
                convo_cheat_sheet[tm_utils.LAST_USER],
            )
        else:
            # ASSISTANT then USER mode [Second pass]
            episode_idx -= len(
                self.episode_map["U"]
            )  # Didn't use '%' because the two maybe unequal in length
            true_idx = self.episode_map["A"][episode_idx]
            conversation = self.convos[true_idx]
            convo_cheat_sheet = self.ep_cheat_sheet[true_idx]
            first_entry, last_entry = (
                convo_cheat_sheet[tm_utils.FIRST_ASSISTANT],
                convo_cheat_sheet[tm_utils.LAST_ASSISTANT],
            )

        starts_at_odd = first_entry % 2 != 0
        if starts_at_odd:
            predecessor = conversation[entry_idx * 2 + 1]['text']
            successor = conversation[entry_idx * 2 + 2]['text']
            ep_done = entry_idx * 2 + 1 == last_entry
        else:
            predecessor = conversation[entry_idx * 2]['text']
            successor = conversation[entry_idx * 2 + 1]['text']
            ep_done = entry_idx * 2 == last_entry

        action = {
            'id': self.id,
            'text': predecessor,
            'episode_done': ep_done,
            'labels': [successor],
        }

        return action


class SelfDialogueSegmentTeacher(FixedDialogTeacher):
    """
    Teacher for written two-person dialogues with labels being relevant/useful
    parts in the input sentence. The different datatypes of the labels within
    the data have also been encoded as `label_types`
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        opt['fn'] = "self-dialogs.json"

        if 'exclude-invalid-data' not in opt:
            opt['exclude-invalid-data'] = True

        if shared and 'convos' in shared:
            # another instance was set up already, just reference its data
            self.convos = shared['convos']
        else:
            # need to set up data from scratch
            data_path = tm_utils._path(opt)
            self._setup_data(data_path, opt)

        self.reset()

    def get(self, episode_idx, entry_idx):
        conversation = self.convos[episode_idx]
        conv_len = len(conversation['utterances'])
        utterance = conversation['utterances'][entry_idx]['text']

        # Check if episode is complete
        ep_done = entry_idx == conv_len - 1
        action = {'id': self.id, 'text': utterance, 'episode_done': ep_done}

        # Setup Labels as "text" from segments
        action['labels'] = []
        action['label_types'] = []
        segments = conversation['utterances'][entry_idx]["segments"]
        for segment in segments:
            action['labels'] += [segment["text"]]
            tmp = []
            for annot in segment["annotations"]:
                tmp += [annot["name"]]
            action['label_types'] += [tmp]

        return action

    def num_examples(self):
        ctr = 0
        for convo in self.convos:
            ctr += len(convo["utterances"])
        return ctr

    def num_episodes(self):
        return len(self.convos)

    def _setup_data(self, data_path, opt):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.convos = json.load(data_file)

        # Filter out instances which do not have "segment" in them
        convos_updated = []
        for convo in self.convos:
            updated_dialog = []
            for i in range(len(convo['utterances'])):
                if "segments" in convo['utterances'][i]:
                    updated_dialog += [convo['utterances'][i]]
            convo['utterances'] = updated_dialog
            if convo['utterances']:
                convos_updated += [convo]
        self.convos = convos_updated


class DefaultTeacher(SelfDialogueTeacher):
    pass
