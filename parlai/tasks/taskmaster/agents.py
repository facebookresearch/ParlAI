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
        if 'exclude-invalid-data' not in opt:
            opt['exclude-invalid-data'] = True

        if shared and 'convos' in shared:
            # another instance was set up already, just reference its data
            self.convos = shared['convos']
        else:
            # need to set up data from scratch
            data_path = _path(opt)
            self._setup_data(data_path, opt)

        self.reset()

    def _setup_data(self, data_path, opt):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.convos = json.load(data_file)
        # Pre-processing
        convos_update = []
        self.ep_cheat_sheet = {}
        for convo in self.convos:
            conversation = convo['utterances']
            # Filter out single greet messages
            if len(conversation) > 1:
                self.ep_cheat_sheet[len(self.ep_cheat_sheet)] = gen_ep_cheatsheet(
                    conversation
                )
                convos_update += [conversation]
        self.convos = convos_update

    # add up cheatsheet[4] + cheatsheet[5] for every episode in ep cheatsheet
    def num_examples(self):
        ctr = 0
        for ep in self.ep_cheat_sheet:
            ctr += self.ep_cheat_sheet[ep][4] + self.ep_cheat_sheet[ep][5]
        return ctr

    def num_episodes(self):
        # For two passes over the data: Once to teach USER and once to teach ASSISTANT
        return len(self.convos) * 2

    def get(self, episode_idx, entry_idx):
        if episode_idx < len(self.convos):
            # USER then ASSISTANT mode
            conversation = self.convos[episode_idx]
            ep_done = entry_idx * 2 == self.ep_cheat_sheet[episode_idx][1]
            predecessor = conversation[entry_idx * 2]['text']
            successor = conversation[entry_idx * 2 + 1]['text']
        else:
            # ASSISTANT then USER mode
            episode_idx %= len(self.convos)
            conversation = self.convos[episode_idx]
            ep_done = entry_idx * 2 + 1 == self.ep_cheat_sheet[episode_idx][3]
            predecessor = conversation[entry_idx * 2 + 1]['text']
            successor = conversation[entry_idx * 2 + 2]['text']

        action = {
            'id': self.id,
            'text': predecessor,
            'episode_done': ep_done,
            'labels': [successor],
        }

        return action


class SelfDialogueTeacher(TaskMasterTeacher):
    """
    Teach Written User-Assistant Dialogues
    """

    def __init__(self, opt, shared=None):
        opt['fn'] = "self-dialogs.json"
        super().__init__(opt, shared)


class WozDialogueTeacher(TaskMasterTeacher):
    """
    Teach Spoken Dialogs
    """

    def __init__(self, opt, shared=None):
        opt['fn'] = "woz-dialogs.json"
        super().__init__(opt, shared)

    # def _setup_data(self, data_path, opt):
    #     print('loading: ' + data_path)
    #     with open(data_path) as data_file:
    #         self.convos = json.load(data_file)
    #     # Pre-processing
    #     convos_update = []
    #     x = set()
    #     ctr = 0
    #     # [start_user_idx, end_user_idx, start_assis_idx, end_assis_idx]
    #     self.ep_cheat_sheet = {}
    #     for idx, convo in enumerate(self.convos):
    #         # convo = smoothen_convo(convo, opt)
    #         convo = convo['utterances']
    #         # Filter out single greet messages
    #         if len(convo) > 1:
    #             # if opt['exclude-invalid-data']:
    #             # self.ep_cheat_sheet[idx] = []
    #             # x.add(len(convo))
    #             # if len(convo) == 2:
    #             for i in range(1, len(convo)):
    #                 if convo[i]['speaker'] == convo[i-1]['speaker']:
    #                     print(convo[i])
    #                     print(convo[i - 1])
    #                     # exit()
    #                     ctr += 1
    #             convos_update += [convo]
    #
    #     self.convos = convos_update
    #     print(ctr)
    #     print(len(self.convos))
    #     exit()


class SelfDialogueSegmentTeacher(SelfDialogueTeacher):
    """
    Teach "self-dialogs.json" with segment texts as labels
    """

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
        # assert(len(action['labels']) == len(action['label_types']))

        return action

    def num_examples(self):
        ctr = 0
        for convo in self.convos:
            ctr += len(convo["utterances"])
        return ctr

    def _setup_data(self, data_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.convos = json.load(data_file)

        # Filter out instances which do not have "segment" in them
        convos_updated = []
        for convo in self.convos:
            updated_dialog = []
            for i in range(0, len(convo['utterances'])):
                if "segments" in convo['utterances'][i]:
                    updated_dialog += [convo['utterances'][i]]
            convo['utterances'] = updated_dialog
            if convo['utterances']:
                convos_updated += [convo]
        self.convos = convos_updated


# Utils
def _path(opt):
    # ensure data is built
    build(opt)
    return os.path.join(opt['datapath'], 'taskmaster-1', opt['fn'])


# Generate a cheatsheet for an episode
def gen_ep_cheatsheet(convo):
    # passed in: utterances
    cheatsheet = [-1, -1, -1, -1, -1, -1]
    # Assumed that length of convo is greater than two due to filtering cond
    for idx in range(1, len(convo)):
        # find first USER with reply
        if convo[idx - 1]['speaker'] == "USER" and convo[idx]['speaker'] == "ASSISTANT":
            if cheatsheet[0] == -1:
                cheatsheet[0] = idx - 1
            # find last USER with reply
            cheatsheet[1] = idx - 1
        # find first ASSISTANT with reply
        if convo[idx - 1]['speaker'] == "ASSISTANT" and convo[idx]['speaker'] == "USER":
            if cheatsheet[2] == -1:
                cheatsheet[2] = idx - 1
            # find last ASSISTANT with reply
            cheatsheet[3] = idx - 1
        # Calculate number of user examples
        cheatsheet[4] = (cheatsheet[1] - cheatsheet[0]) // 2 + 1
        # Calculate number of assistant examples
        cheatsheet[5] = (cheatsheet[1] - cheatsheet[0]) // 2 + 1

    return cheatsheet


# Re-assign indexes after smoothening (mostly for clarity purposes)
# Doesn't matter since we never index by specifically using the index field of the json
def update_indexes(conversation):
    for i in range(len(conversation)):
        conversation[i]["index"] = i

    return conversation


# Join two conversations
# Join texts don't care about segments
# Assumption: utt1 is the one popped from the stack
def join_speech(utt1, utt2):
    new_utt = {}
    new_utt["index"] = utt1["index"]
    new_utt["text"] = utt1["text"] + "\n" + utt2["text"]
    new_utt["speaker"] = utt1["speaker"]
    if 'ctr' in utt1:
        new_utt['ctr'] = utt1['ctr'] + 1
    else:
        new_utt['ctr'] = 2
    return new_utt


# Aggregate contiguous responses by the same speaker in the data
def smoothen_convo(conversation, opt):
    dialogue = conversation['utterances']
    conversation_stack = []
    for speech in dialogue:
        if (
            conversation_stack
            and speech["speaker"] == conversation_stack[-1]["speaker"]
        ):
            conversation_stack.append(join_speech(conversation_stack.pop(), speech))
        else:
            conversation_stack.append(speech)
    processed_conversation = []
    for speech in conversation_stack:
        if opt['exclude-invalid-data'] and 'ctr' in speech and speech['ctr'] > 5:
            continue
        else:
            processed_conversation += [speech]
    return update_indexes(processed_conversation)
