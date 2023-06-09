#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Teacher for the public chatbot data.
"""

import os
import json
from copy import deepcopy

from parlai.core.message import Message
from parlai.utils.data import DatatypeHelper
from parlai.core.teachers import ChunkTeacher
from .build import build


TOTAL_NUM_CHUNKS = 100
NUM_TEST_CHUNKS = 3
NUM_VALID_CHUNKS = 3
NUM_TRAIN_CHUNKS = 94

BOT_AGENT_LABEL = 'bot'
HUMAN_AGENT_LABEL = 'human'

TRAIN = "train"
VALID = "valid"
TEST = "test"

SAFETY_OK_VALUE = '__ok__'


class Teachers:
    BB3_DATA_BOT_TEACHER = "bb3-data-bot-teacher"
    BB3_DATA_HUMAN_TEACHER = "bb3-data-human-teacher"
    BB3_DATA_FILTER_ADVERSARIAL_HUMANS_HUMAN_TEACHER = (
        "bb3-data-filter-adversarial-humans-human-teacher"
    )
    BB3_DATA_FILTER_ADVERSARIAL_HUMANS_BOT_TEACHER = (
        "bb3-data-filter-adversarial-humans-bot-teacher"
    )
    BB3_DATA_CROWDWORKERS_BOT_TEACHER = "bb3-data-crowdworkers-bot-teacher"
    BB3_DATA_CROWDWORKERS_HUMAN_TEACHER = "bb3-data-crowdworkers-human-teacher"


LABEL_PER_TEACHER = {'bb3-data-bot-teacher': 'bot', 'bb3-data-human-teacher': 'human'}

NUM_SAMPLES = {
    Teachers.BB3_DATA_HUMAN_TEACHER: {
        TRAIN: (2605234, 2605234),
        VALID: (81291, 81291),
        TEST: (82336, 82336),
    },
    Teachers.BB3_DATA_BOT_TEACHER: {
        TRAIN: (2682397, 2682397),
        VALID: (84016, 84016),
        TEST: (85127, 85127),
    },
    Teachers.BB3_DATA_FILTER_ADVERSARIAL_HUMANS_HUMAN_TEACHER: {
        TRAIN: (752566, 752566),
        VALID: (22898, 22898),
        TEST: (21924, 21924),
    },
    Teachers.BB3_DATA_FILTER_ADVERSARIAL_HUMANS_BOT_TEACHER: {
        TRAIN: (773828, 773828),
        VALID: (23635, 23635),
        TEST: (22759, 22759),
    },
    Teachers.BB3_DATA_CROWDWORKERS_BOT_TEACHER: {
        TRAIN: (14324, 14324),
        VALID: (567, 567),
        TEST: (582, 582),
    },
    Teachers.BB3_DATA_CROWDWORKERS_HUMAN_TEACHER: {
        TRAIN: (6672, 6672),
        VALID: (268, 268),
        TEST: (232, 232),
    },
}


def get_dtype(opt):
    return DatatypeHelper.fold(opt['datatype'])


class BaseBB3DataTeacher(ChunkTeacher):
    """
    Do NOT use this directly, use its children.
    """

    def __init__(self, opt, shared=None):
        build(opt)
        self.opt = deepcopy(opt)
        self.id = "BaseBB3DataTeacher"
        self.dpath = os.path.join(self.opt['datapath'], 'bb3_demo/release_data_chunks')
        super().__init__(opt, shared=shared)

    def _get_data_folder(self):
        """
        return the path to directory containing your chunks.
        """
        return self.dpath

    def _get_fname(self, chunk_idx: int) -> str:
        """
        Get the filename of the data chunk.

        :param chunk_idx:
            which chunk to get
        :return chunk_name:
            return the chunk fname
        """
        return f'release_data_{str(chunk_idx).zfill(2)}.jsonl'

    def get_num_samples(self, opt):
        dt = get_dtype(opt)
        return NUM_SAMPLES[self._teacher_type()][dt]

    def get_fold_chunks(self, opt):
        datatype = get_dtype(opt)
        if "test" == datatype:
            s = 0
            e = s + NUM_TEST_CHUNKS
        elif "valid" == datatype:
            s = NUM_TEST_CHUNKS
            e = s + NUM_VALID_CHUNKS
        elif "train" == datatype:
            s = NUM_TEST_CHUNKS + NUM_VALID_CHUNKS
            e = s + NUM_TRAIN_CHUNKS

        return list(range(s, e))

    def _teacher_type(self):
        """
        For determining the key to read the NUM_SAMPLES.
        """
        raise NotImplementedError(
            "Teachers inheriting from BaseBB3DataTeacher "
            "must implement their own _teacher_type function."
        )

    def _partner_agent(self, agent):
        """
        Determine identifier of partner.
        """
        if agent == BOT_AGENT_LABEL:
            return HUMAN_AGENT_LABEL
        else:
            return BOT_AGENT_LABEL

    def _is_safe(self, message):
        # based on safety classifier used to annotate data
        safety_res = message.get('safety')
        return (
            safety_res['duo_safety'] == SAFETY_OK_VALUE
            and safety_res['string_matcher'] == SAFETY_OK_VALUE
        )

    def _filter_out_message(self, message, message_history):
        """
        Whether to exclude a message based on a certain criteria.
        """
        return False

    def _examples_from_convo(self, convo, label_agent):
        """
        Bot teacher: text = human message and label = bot response
        Human teacher:  text = bot message and label = human response
        """
        examples = []
        message_history = convo.pop('message_history')
        # adversarial human = human that triggered safety classifier at least once in convo
        adversarial_human = any(
            [m['sender'] == 'human' and not self._is_safe(m) for m in message_history]
        )
        for idx, m in enumerate(message_history):
            if self._filter_out_message(m, message_history):
                continue
            if m['sender'] == label_agent:
                context = message_history[:idx]
                text_context = [d['text'] for d in context]
                convo_metadata_fields = convo
                # if no context (1st message), skip
                if len(text_context) == 0:
                    continue
                parlai_msg = Message(
                    {
                        **convo_metadata_fields,
                        'text': '\n'.join(text_context),
                        'labels': [m['text']],
                        'label_info': m,
                        'episode_done': True,
                        'adversarial_human': adversarial_human,
                    }
                )
                examples.append(parlai_msg)
        return examples

    def _examples(self, convo):
        return self._examples_from_convo(convo, LABEL_PER_TEACHER[self._teacher_type()])

    def load_from_chunk(self, chunk_idx: int):
        # we load the chunk specified by chunk_idx and return a
        # list of examples
        fname = self._get_fname(chunk_idx)
        chunk_path = os.path.join(self._get_data_folder(), fname)
        output = []
        with open(chunk_path, 'r') as json_file:
            json_list = list(json_file)
            for json_str in json_list:
                convo = json.loads(json_str)
                examples = self._examples(convo)
                output += examples
        return output

    def create_message(self, example, entry_idx=0):
        return example


class CrowdworkersBaseTeacher(BaseBB3DataTeacher):
    """
    Only loads examples where label has crowdworker annotations.

    Do NOT use directly.
    """

    def _filter_out_message(self, message, message_history):
        return message.get('mturk', None) is None


class BB3DataBotTeacher(BaseBB3DataTeacher):
    """
    Basic dialogue teacher where label is bot.
    """

    def __init__(self, opt, shared=None):
        self.id = "BB3DataBotTeacher"
        super().__init__(opt, shared=shared)

    def _teacher_type(self):
        return Teachers.BB3_DATA_BOT_TEACHER


class BB3DataHumanTeacher(BaseBB3DataTeacher):
    """
    Basic dialogue teacher where label is human.
    """

    def __init__(self, opt, shared=None):
        self.id = "BB3DataHumanTeacher"
        super().__init__(opt, shared=shared)

    def _teacher_type(self):
        return Teachers.BB3_DATA_HUMAN_TEACHER


class FilterOutAdversarialHumansBaseTeacher(BaseBB3DataTeacher):
    """
    Filters out examples from users who trigger safety classifier at all throughout
    conversation.

    Do NOT use directly.
    """

    def _filter_out_message(self, label_message, message_history):
        adversarial_human = any(
            [m['sender'] == 'human' and not self._is_safe(m) for m in message_history]
        )
        return adversarial_human


class FilterOutAdversarialHumansHumanTeacher(
    BB3DataHumanTeacher, FilterOutAdversarialHumansBaseTeacher
):
    """
    Filters out examples from users who trigger safety classifier at all throughout
    conversation.
    """

    def get_num_samples(self, opt):
        dt = get_dtype(opt)
        return NUM_SAMPLES[Teachers.BB3_DATA_FILTER_ADVERSARIAL_HUMANS_HUMAN_TEACHER][
            dt
        ]


class FilterOutAdversarialHumansBotTeacher(
    BB3DataBotTeacher, FilterOutAdversarialHumansBaseTeacher
):
    """
    Filters out examples from users who trigger safety classifier at all throughout
    conversation.
    """

    def get_num_samples(self, opt):
        dt = get_dtype(opt)
        return NUM_SAMPLES[Teachers.BB3_DATA_FILTER_ADVERSARIAL_HUMANS_BOT_TEACHER][dt]


class BB3DataCrowdworkersBotTeacher(BB3DataBotTeacher, CrowdworkersBaseTeacher):
    """
    Label is bot and has crowdworker annotations.
    """

    def get_num_samples(self, opt):
        dt = get_dtype(opt)
        return NUM_SAMPLES[Teachers.BB3_DATA_CROWDWORKERS_BOT_TEACHER][dt]


class BB3DataCrowdworkersHumanTeacher(BB3DataHumanTeacher, CrowdworkersBaseTeacher):
    """
    Label is human and has crowdworker annotations.
    """

    def get_num_samples(self, opt):
        dt = get_dtype(opt)
        return NUM_SAMPLES[Teachers.BB3_DATA_CROWDWORKERS_HUMAN_TEACHER][dt]


class DefaultTeacher(BB3DataBotTeacher):
    pass
