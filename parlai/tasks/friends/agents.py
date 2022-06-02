#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.core.params import ParlaiParser
from .build import build

import copy
import os
import json


def _path(opt, *additions):
    return os.path.join(
        opt['datapath'], 'Friends', 'friends-corpus/utterances.jsonl', *additions
    )


class DefaultTeacher(DialogTeacher):
    START_TOKEN = '<START>'
    SILENCE_TOKEN = '<SILENCE>'
    MAIN_CHARACTERS = [
        'Rachel Green',
        'Monica Geller',
        'Phoebe Buffay',
        'Joey Tribbiani',
        'Chandler Bing',
        'Ross Geller',
    ]

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        build(opt)
        opt['datafile'] = _path(opt)
        self.character = opt['character']
        self.use_silence_token = opt['use_silence_token']
        self.use_start_token = opt['use_start_token']
        super().__init__(opt, shared)

    def setup_data(self, datafile):
        conversations = {}

        with open(datafile, 'r') as json_file:
            for json_str in json_file:
                utterance = json.loads(json_str)

                text = utterance['text']
                speaker = utterance['speaker']
                conversation_id = utterance['conversation_id']

                if conversation_id not in conversations:
                    conversations[conversation_id] = []
                conversations[conversation_id].append(
                    {"text": text, "speaker": speaker}
                )

        for conversation in conversations:
            utterances = conversations[conversation]
            last_utterance_index = len(utterances) - 1

            for index, utterance in enumerate(utterances):
                if index == 0:
                    if self.use_start_token:
                        context = self.START_TOKEN

                    else:  # skip the first utterance since there's no context
                        speaker = utterance['speaker']
                        text = utterance['text']
                        context = f'{speaker}: {text}'
                        continue

                speaker = utterance['speaker']
                text = utterance['text']

                prev_context = context
                context += '\n' + f'{speaker}: {text}'

                isConversationDone = index == last_utterance_index

                # By default, generate training examples for all 6 main characters.
                # Otherwise only generate training examples for the chosen character.
                if (
                    self.character == 'All' and speaker in self.MAIN_CHARACTERS
                ) or speaker == self.character:
                    yield {"text": prev_context, "label": text}, isConversationDone
                elif self.use_silence_token:
                    yield {
                        "text": prev_context,
                        "label": self.SILENCE_TOKEN,
                    }, isConversationDone

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Friends Corpus Arguments')
        agent.add_argument(
            '--character',
            type=str,
            default='All',
            choices=[
                'All',
                'Rachel Green',
                'Monica Geller',
                'Phoebe Buffay',
                'Joey Tribbiani',
                'Chandler Bing',
                'Ross Geller',
            ],
            help='Which speaker labels to train on',
        )
        agent.add_argument(
            '--use_silence_token',
            type='bool',
            default=True,
            help='Use silence token <SILENCE> to generate training example for sentences where the chosen speaker is not speaking',
        )
        agent.add_argument(
            '--use_start_token',
            type='bool',
            default=False,
            help='Use start token <START> at the beginning of each conversation, and include the first sentence as a training example',
        )
        return parser
