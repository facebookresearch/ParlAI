#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .build import build
import os

# Constants
FIRST_USER = 0
LAST_USER = 1
FIRST_ASSISTANT = 2
LAST_ASSISTANT = 3
USER_NUM_EX = 4
ASSIS_NUM_EX = 5


# Functions
def _path(opt):
    # ensure data is built
    build(opt)
    return os.path.join(opt['datapath'], 'taskmaster-1', opt['fn'])


# Generate a cheatsheet for an episode
def gen_ep_cheatsheet(convo):
    # passed in: utterances
    cheatsheet = [-1, -1, -1, -1, 0, 0]
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

        # Calculate num_examples only if atleast one user was found
        if cheatsheet[1] != -1:
            # Calculate number of user examples
            cheatsheet[4] = (cheatsheet[1] - cheatsheet[0]) // 2 + 1
        # Calculate num_examples only if atleast one user was found
        if cheatsheet[3] != -1:
            # Calculate number of assistant examples
            cheatsheet[5] = (cheatsheet[3] - cheatsheet[2]) // 2 + 1

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
    corrupt = False
    for speech in conversation_stack:
        if opt['exclude-invalid-data'] and 'ctr' in speech and speech['ctr'] > 5:
            corrupt = True
        processed_conversation += [speech]
    return update_indexes(processed_conversation), corrupt
