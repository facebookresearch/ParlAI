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
import random
import copy
import os

RANDOM_SEED = 123
random.seed(RANDOM_SEED)

START_TOKEN = '__START__'
SILENCE_TOKEN = '__SILENCE__'


def _path(opt, filename):
    return os.path.join(opt['datapath'], 'Friends', filename)


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        build(opt)
        self.fold = DatatypeHelper.fold(opt['datatype'])
        opt['datafile'] = _path(opt, self.fold + '.jsonl')
        self.characters = opt['characters'].split(',')
        self.character = opt['character']
        self.include_speaker_in_context = opt['include_speaker_in_context']
        self.add_speaker_to_context_end = opt['add_speaker_to_context_end']
        self.silence_token_dropout = opt['silence_token_dropout']
        self.silence_token = opt['silence_token']
        self.use_start_token = opt['use_start_token']
        self.start_token = opt['start_token']
        self.utterance_delimiter = opt['utterance_delimiter']
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

        for conversation_id in conversations:
            utterances = conversations[conversation_id]
            characters = set(
                [u['speaker'] for u in utterances if u['speaker'] in self.characters]
            )
            characters_string = ','.join(
                sorted(list(characters))
            )  # sorted to ensure same order across runs
            last_utterance_index = len(utterances) - 1
            speakers = []

            for index, utterance in enumerate(utterances):
                if index == 0:
                    if self.use_start_token:
                        context = self.start_token

                    else:  # skip the first utterance since there's no context
                        speaker = utterance['speaker']
                        speakers.append(speaker)

                        text = self._get_text(utterance)
                        if self.include_speaker_in_context:
                            context = f'{speaker}: {text}'
                        else:
                            context = text
                        continue

                speaker = utterance['speaker']

                text = self._get_text(utterance)
                prev_context = context
                if self.include_speaker_in_context:
                    context += self.utterance_delimiter + f'{speaker}: {text}'
                else:
                    context += self.utterance_delimiter + text

                isConversationDone = index == last_utterance_index

                # By default, generate training examples for all 6 main characters.
                # Otherwise only generate training examples for the chosen character.
                if (
                    self.character == 'All' and speaker in self.characters
                ) or speaker == self.character:
                    text, label, speakers, hasAddedSpeaker = self._get_message_fields(
                        text, speaker, speakers, prev_context
                    )
                    _speakers = speakers[:]
                    if not hasAddedSpeaker:
                        speakers.append(speaker)
                    yield {
                        "text": text,
                        "label": label,
                        "characters": characters_string,
                        "speakers": _speakers,
                    }, isConversationDone
                elif random.random() > self.silence_token_dropout:
                    text, label, speakers, hasAddedSpeaker = self._get_message_fields(
                        self.silence_token, self.character, speakers, prev_context
                    )
                    _speakers = speakers[:]
                    if not hasAddedSpeaker:
                        speakers.append(speaker)
                    yield {
                        "text": text,
                        "label": label,
                        "characters": characters_string,
                        "speakers": _speakers,
                    }, isConversationDone
                else:
                    speakers.append(speaker)

    def _get_text(self, utterance):
        """
        Replace newline character by whitespace so that the data format plays nicely
        with BB2, which splits each utterance by newline and expects a corresponding
        speaker label (if we don't replace the newline character here, we have to later
        match each speaker label back to variable number of sentences, which overly
        complicates things) c.f.

        line 606 of projects/blenerbot2/agents/blenderbot2.py
        """
        return utterance['text'].replace('\n', ' ')

    def _get_message_fields(self, text, speaker, speakers, prev_context):
        """
        If `include_speaker_in_context` is True, keep speaker ids in the text.

        If `add_speaker_to_context_end` is True, add speaker ids at the end of text, and
        remove speaker ids from the labels. If `include_speaker_in_context` is False,
        but `add_speaker_to_context_end` is True, add an empty sentence at the end of
        text and add the current speaker id to the list of speakers, to indicate the
        speaker for the empty sentence.
        """
        hasAddedSpeaker = False
        if self.include_speaker_in_context:
            if self.add_speaker_to_context_end:
                label = text
                text = prev_context + f'{self.utterance_delimiter}{speaker}: '
                # Save current spaker as the speaker for the empty utterance
                speakers.append(speaker)
                hasAddedSpeaker = True
            else:
                label = f'{speaker}: {text}'
                text = prev_context
        else:
            if self.add_speaker_to_context_end:
                label = text
                # The whitespace is left at the end to indicate an empty utterance
                text = prev_context + f'{self.utterance_delimiter} '
                # Save current spaker as the speaker for the empty utterance
                speakers.append(speaker)
                hasAddedSpeaker = True
            else:
                label = f'{speaker}: {text}'
                text = prev_context

        return text, label, speakers, hasAddedSpeaker

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
            '--characters',
            type=str,
            default='Rachel Green,Monica Geller,Phoebe Buffay,Joey Tribbiani,Chandler Bing,Ross Geller',
            help='A comma-separated list of characters to train on when `--character` == `All`',
        )
        agent.add_argument(
            '--utterance-delimiter',
            type=str,
            default='\n',
            help="A string used to separate each utterance in the context. Defaults to newline. For example, 'A: Hello\nB: Hi there'.",
        )
        agent.add_argument(
            '--include-speaker-in-context',
            type='bool',
            default=True,
            help="Whether to include speaker labels in the context. For example, message = { text: 'Rachel: Hi' } instead of message = { text: 'Hi' }",
        )
        agent.add_argument(
            '--add-speaker-to-context-end',
            type='bool',
            default=True,
            help='Append speaker to the end of each context. Defaults to True.',
        )
        agent.add_argument(
            '--silence-token-dropout',
            type=float,
            default=1,
            help='Dropout probability for using silence token to generate training example for sentences where the chosen speaker is not speaking. When set to 0, all silence tokens will generate training examples. When set to 1, no silence tokens will generate training examples. Defaults to 1.',
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
        opt['character'] = 'All'
        super().__init__(opt, shared)


class RachelTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['character'] = 'Rachel Green'
        super().__init__(opt, shared)


class MonicaTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['character'] = 'Monica Geller'
        super().__init__(opt, shared)


class PhoebeTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['character'] = 'Phoebe Buffay'
        super().__init__(opt, shared)


class JoeyTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['character'] = 'Joey Tribbiani'
        super().__init__(opt, shared)


class ChandlerTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['character'] = 'Chandler Bing'
        super().__init__(opt, shared)


class RossTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['character'] = 'Ross Geller'
        super().__init__(opt, shared)
