#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility methods for conversations format.

"""
import datetime
import json
import os
import random


class Metadata:
    """
    Utility class for conversation metadata.

    Metadata should be saved at <datapath>.metadata.
    """
    def __init__(self, datapath):
        self._load(datapath)

    def _load(self, datapath):
        metadata_path = self._get_path(datapath)
        if not os.path.isfile(metadata_path):
            raise RuntimeError(
                f'Metadata at path {metadata_path} not found. '
                'Double check your path.'
            )

        with open(metadata_path, 'rb') as f:
            metadata = json.load(f)

        self.datetime = metadata['date']
        self.opt = metadata['opt']
        self.self_chat = metadata['self_chat']
        self.speaker_1 = metadata['speaker_1']
        self.speaker_2 = metadata['speaker_2']
        self.extra_data = {}
        for k, v in metadata.items():
            if k not in ['date', 'opt', 'speaker_1', 'speaker_2']:
                self.extra_data[k] = v

    def read(self):
        """
        Read the relevant metadata
        """
        print(f'Saved at: {self.datetime}')
        print(f'Self chat: {self.self_chat}')
        print(f'Speaker 1: {self.speaker_1}')
        print(f'Speaker 2: {self.speaker_2}')
        print('Opt:')
        for k, v in self.opt.items():
            print(f'\t{k}: {v}')
        for k, v in self.extra_data.items():
            print(f'{k}: {v}')

    @staticmethod
    def _get_path(datapath):
        fle, _ = os.path.splittext(datapath)
        return os.path.join(fle, '.metadata')

    @classmethod
    def save_metadata(
        cls,
        datapath,
        opt,
        self_chat=False,
        speaker_1='human',
        speaker_2=None,
        **kwargs,
    ):
        """
        Dump conversation metadata to file.
        """
        metadata = {}
        metadata['date'] = str(datetime.datetime.now())
        metadata['opt'] = opt
        metadata['self_chat'] = self_chat
        metadata['speaker_1'] = speaker_1
        if speaker_2 is not None:
            metadata['speaker_2'] = speaker_2
        else:
            metadata['speaker_2'] = opt.get('model')

        for k, v in kwargs.items():
            metadata[k] = metadata[v]

        metadata_path = cls._get_path(datapath)
        print(f'[ Writing metadata to file {metadata_path} ]')
        with open(metadata_path, 'w') as f:
            f.write(json.dumps(metadata))


class Conversations:
    """
    Utility class for reading and writing from ParlAI Conversations format.

    Conversations should be saved in JSONL format, where each line is
    a JSON of the following form:
    {
        'possible_conversation_level_info': True,
        'dialogue':
            [
                {
                    'id': 'speaker_1',
                    'text': <first utterance>,
                },
                {
                    'id': 'speaker_2',
                    'text': <second utterance>,
                },
            ]
        ...
    }
    """
    def __init__(self, datapath):
        self.conversations = self._load_conversations(datapath)
        self.metadata = self._load_metadata(datapath)

    @property
    def num_conversations(self):
        return(len(self.conversations))

    def _load_conversations(self, datapath):
        if not os.path.isfile(datapath):
            raise RuntimeError(
                f'Conversations at path {datapath} not found. '
                'Double check your path.'
            )

        conversations = []
        with open(datapath, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                conversations.append(json.loads(line))

        return conversations

    def _load_metadata(self, datapath):
        """
        Load metadata.

        Metadata should be saved at <identifier>.metadata
        Metadata should be of the following format:
        {
            'date': <date collected>,
            'opt': <opt used to collect the data,
            'speaker_1': <identity of speaker 1>,
            'speaker_2': <identify of speaker 2>,
            ...
            Other arguments.
        }

        """
        try:
            metadata = Metadata(datapath)
            return metadata
        except RuntimeError:
            print(
                'Metadata does not exist. Please double check '
                'your datapath.'
            )
            return None

    def read_metadata(self):
        if self.metadata is not None:
            self.metadata.read()

    def read_conv_idx(self, idx):
        convo = self.conversations[idx]
        print('=' * 75)

        high_level = [k for k in convo.keys() if k != 'dialogue']
        if high_level:
            for key in high_level:
                print(f'{key}: {convo[key]}')
            print('-' * 75)

        for turn in convo:
            turn_id = turn['id']
            text = turn['text']
            print(f'{turn_id}: {text}')

        print('=' * 75)

    def read_rand_conv(self):
        idx = random.choice(range(self.num_conversations))
        self.read_conv_idx(idx)
