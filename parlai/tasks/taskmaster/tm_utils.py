#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .build import build
import os

###############################################################
#                                                             #
# Constants                                                   #
#                                                             #
###############################################################
FIRST_USER_IDX = 0
LAST_USER_IDX = 1
FIRST_ASSISTANT_IDX = 2
LAST_ASSISTANT_IDX = 3
USER_NUM_EX = 4
ASSIS_NUM_EX = 5


###############################################################
#                                                             #
# Functions                                                   #
#                                                             #
###############################################################
def _path(opt):
    """
    Ensures that the data is build and returns path to specific data file.

    :param opt:
        Options dict: mainly used to access the data file name while path creation.
    """
    # ensure data is built
    build(opt)
    return os.path.join(opt['datapath'], 'taskmaster-1', opt['fn'])


def gen_ep_cheatsheet(convo):
    """
    Generates a cheatsheet for a particular conversation (episode).
    The index and it's significance in the cheatsheet is shown below:
        0: First index of a USER that has an ASSISTANT reply to it
        1: Last index of a USER that has an ASSISTANT reply to it
        2: First index of an ASSISTANT that has a USER reply to it
        3: Last index of an ASSISTANT that has a USER reply to it
        4: Number of examples for USER speech  as text and ASSISTANT speech as label
        5: Number of examples for ASSISTANT speech as text and USER speech  as label
    :param convo:
        The dialogue between USER and ASSISTANT [after smoothening]
    """
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


def update_indexes(conversation):
    """
    Re-assigns indexes after smoothening (mostly for clarity purposes) Doesn't really
    matter since we never index by specifically using the "index" field of the json obj.

    :param conversation:
        The dialogue between USER and ASSISTANT with inconsistent indices
    """
    for i in range(len(conversation)):
        conversation[i]["index"] = i

    return conversation


def join_speech(utt1, utt2):
    """
    Joins two conversations using a '\n'
    Join texts and doesn't care about segments
    Assumption: utt1 is the one being popped off from the stack and the utt2
                is the one trying to be added. This is so that I check for 'ctr'
                field only in one of the parameters.
    :param utt1:
        An utterance (json object) from either USER or ASSISTANT
    :param utt2:
        Next utterance(json object) from a speaker same as utt1
    """
    new_utt = {}
    new_utt["index"] = utt1["index"]
    new_utt["text"] = utt1["text"] + "\n" + utt2["text"]
    new_utt["speaker"] = utt1["speaker"]
    if 'ctr' in utt1:
        new_utt['ctr'] = utt1['ctr'] + 1
    else:
        new_utt['ctr'] = 2
    return new_utt


def smoothen_convo(conversation, opt):
    """
    Aggregates contiguous responses by the same speaker in the data so that data
    eventually contains alternations between USER and ASSISTANT.

    :param conversation:
        The dialogue between USER and ASSISTANT with possible multiple contiguous
        speeches by the same speaker
    :param opt:
        options dict, mainly useful for accessing value of exclude_invalid_data
    """
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
        if (
            opt.get('exclude_invalid_data', True)
            and 'ctr' in speech
            and speech['ctr'] > 5
        ):
            corrupt = True
        processed_conversation += [speech]
    return update_indexes(processed_conversation), corrupt
