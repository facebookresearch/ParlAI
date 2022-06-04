#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.core.params import ParlaiParser
from .build import build
from collections import defaultdict
import jsonlines
from parlai.utils.data import DatatypeHelper

import copy
import os

START_TOKEN = '__START__'
SILENCE_TOKEN = '__SILENCE__'


def _path(opt, filename):
    return os.path.join(opt['datapath'], 'Friends', filename)


class DefaultTeacher(DialogTeacher):

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
        self.fold = DatatypeHelper.fold(opt['datatype'])
        opt['datafile'] = _path(opt, self.fold + '.jsonl')
        self.character = opt['character']
        self.use_silence_token = opt['use_silence_token']
        self.silence_token = opt['silence_token']
        self.use_start_token = opt['use_start_token']
        self.start_token = opt['start_token']
        super().__init__(opt, shared)

    def setup_data(self, datafile):
        conversations = defaultdict(list)

        with jsonlines.open(datafile) as reader:
            for utterance in reader:
                text = utterance['text']
                speaker = utterance['speaker']
                conversation_id = utterance['conversation_id']

                conversations[conversation_id].append(
                    {"text": text, "speaker": speaker}
                )

        for conversation in conversations:
            utterances = conversations[conversation]
            last_utterance_index = len(utterances) - 1

            for index, utterance in enumerate(utterances):
                if index == 0:
                    if self.use_start_token:
                        context = self.start_token

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
                    yield {
                        "text": prev_context,
                        "label": f'{speaker}: {text}',
                    }, isConversationDone
                elif self.use_silence_token:
                    yield {
                        "text": prev_context,
                        "label": f'{self.character}: {self.silence_token}',
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
            '--use-silence-token',
            type='bool',
            default=True,
            help='Use silence token to generate training example for sentences where the chosen speaker is not speaking. Defaults to True.',
        )
        agent.add_argument(
            '--silence-token',
            type=str,
            default=SILENCE_TOKEN,
            help='The token to use to indicate the chosen speaker is silent. Defaults to __SILENCE__',
        )
        agent.add_argument(
            '--use-start-token',
            type='bool',
            default=False,
            help='Use start token at the beginning of each conversation, and include the first sentence as a training example. Defaults to False.',
        )
        agent.add_argument(
            '--start-token',
            type=str,
            default=START_TOKEN,
            help='The token to use to indicate the beginning of a conversation. Defaults to __START__',
        )
        return parser


class AllCharactersTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.character = 'All'


class RachelTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.character = 'Rachel Green'


class MonicaTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.character = 'Monica Geller'


class PhoebeTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.character = 'Phoebe Buffay'


class JoeyTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.character = 'Joey Tribbiani'


class ChandlerTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.character = 'Chandler Bing'


class RossTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.character = 'Ross Geller'
