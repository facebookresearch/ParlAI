#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
import json
import copy


class TaskMasterTeacher(FixedDialogTeacher):
    """
    Base class to define a Teacher for TaskMaster-1 Google 2019
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt)

        # Defaut case (If nothing was set)
        if 'fn' not in opt:
            opt['fn'] = "self-dialogs.json"

        if shared and 'convos' in shared:
            # another instance was set up already, just reference its data
            self.convos = shared['convos']
        else:
            # need to set up data from scratch
            data_path = _path(opt)
            self._setup_data(data_path)

        self.reset()

    def _setup_data(self, data_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.convos = json.load(data_file)
        convos_update = []
        for convo in self.convos:
            # Filter out single greet messages
            if len(convo['utterances']) > 1:
                convos_update += [convo]
        self.convos = convos_update

    # Number of time the assistant speaks
    def num_examples(self):
        ctr = 0
        for convo in self.convos:
            for sentence in convo["utterances"]:
                if sentence["speaker"] == "ASSISTANT":
                    ctr += 1
        return ctr

    def num_episodes(self):
        return len(self.convos)

    def get(self, episode_idx, entry_idx):
        conversation = self.convos[episode_idx]
        conv_len = len(conversation['utterances'])
        predecessor = conversation['utterances'][entry_idx]['text']
        successor = conversation['utterances'][entry_idx + 1]['text']

        # Check if episode is complete
        ep_done = False
        if entry_idx + 1 == conv_len - 1:
            ep_done = True

        action = {'id': self.id, 'text': predecessor, 'episode_done': ep_done}
        action['labels'] = [successor]

        return action


class SelfDialogueTeacher(TaskMasterTeacher):
    """
    Teach self-dialogs.json
    """

    def __init__(self, opt, shared=None):
        opt['fn'] = "self-dialogs.json"
        super().__init__(opt, shared)


class WozDialogueTeacher(TaskMasterTeacher):
    """
    Teach woz-dialogs.json
    """

    def __init__(self, opt, shared=None):
        opt['fn'] = "woz-dialogs.json"
        super().__init__(opt, shared)


class SelfDialogueSegmentTeacher(SelfDialogueTeacher):
    """
    Teach "self-dialogs.json" with segment texts as labels
    """

    def get(self, episode_idx, entry_idx):
        conversation = self.convos[episode_idx]
        conv_len = len(conversation['utterances'])
        utterance = conversation['utterances'][entry_idx]['text']

        # Check if episode is complete
        ep_done = False
        if entry_idx == conv_len - 1:
            ep_done = True

        action = {'id': self.id, 'text': utterance, 'episode_done': ep_done}

        # Setup Labels as "text" from segments
        action['labels'] = []
        segments = conversation['utterances'][entry_idx]["segments"]
        for segment in segments:
            action['labels'] += [segment["text"]]

        return action

    def _setup_data(self, data_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.convos = json.load(data_file)

        # Filter out instances which do not have "segment" in them
        convos_updated = []
        for convo in self.convos:
            convo_copy = copy.deepcopy(convo['utterances'])
            updated_dialog = []
            for i in range(0, len(convo_copy)):
                if "segments" in convo_copy[i]:
                    updated_dialog += [convo_copy[i]]
            convo['utterances'] = updated_dialog
            if convo['utterances']:
                convos_updated += [convo]
        self.convos = convos_updated


# Utils
def _path(opt):
    # ensure data is built
    build(opt)
    return os.path.join(opt['datapath'], 'taskmaster-1', opt['fn'])
