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


BAR = '=' * 60
SMALL_BAR = '-' * 60


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
            if k not in ['date', 'opt', 'speaker_1', 'speaker_2', 'self_chat']:
                self.extra_data[k] = v

    def read(self):
        """
        Read the relevant metadata.
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
        fle, _ = os.path.splitext(datapath)
        return fle + '.metadata'

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
        'dialog':
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
        self.iterator_idx = 0

    @property
    def num_conversations(self):
        return len(self.conversations)

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
            print('Metadata does not exist. Please double check your datapath.')
            return None

    def read_metadata(self):
        if self.metadata is not None:
            self.metadata.read()

    def next(self):
        """
        Return the next conversation.
        """
        if self.iterator_idx >= self.num_conversations:
            print('You reached the end of the conversations.')
            self.reset()  # return the iterator idx to 0
            return None

        conv = self.conversations[self.iterator_idx]
        self.iterator_idx += 1

        return conv

    def read_conv_idx(self, idx):
        convo = self.conversations[idx]
        print(BAR)

        high_level = [k for k in convo.keys() if k != 'dialog']
        if high_level:
            for key in high_level:
                print(f'{key}: {convo[key]}')
            print(SMALL_BAR)

        for turn in convo['dialog']:
            turn_id = turn['id']
            text = turn['text']
            print(f'{turn_id}: {text}')

        print(BAR)

    def read_rand_conv(self):
        idx = random.choice(range(self.num_conversations))
        self.read_conv_idx(idx)

    def reset(self):
        self.iterator_idx = 0

    @staticmethod
    def _get_path(datapath):
        fle, _ = os.path.splitext(datapath)
        return fle + '.jsonl'

    @classmethod
    def save_conversations(
        cls,
        act_list,
        datapath,
        opt,
        save_keys='text',
        self_chat=False,
        speaker_1=None,
        speaker_2=None,
        **kwargs,
    ):
        """
        Write Conversations to file from an act list.
        """
        to_save = cls._get_path(datapath)

        # save conversations
        with open(to_save, 'w') as f:
            for ep in act_list:
                if not ep:
                    continue
                convo = {}
                convo['context'] = []
                convo['dialog'] = []
                for act_pair in ep:
                    for i, ex in enumerate(act_pair):
                        ex_id = ex.get('id')
                        # possibly set speaker vars
                        if i == 0 and speaker_1 is None and ex_id is not None:
                            speaker_1 = ex_id
                        elif i == 1 and speaker_2 is None and ex_id is not None:
                            speaker_2 = ex_id

                        # check if act is from speaker 1 or speaker 2
                        context = False
                        if (i % 2 == 0 and ex_id != speaker_1) or (
                            i % 2 == 1 and ex_id != speaker_2
                        ):
                            context = True

                        # set turn
                        turn = {}
                        for key in save_keys.split(','):
                            turn[key] = ex.get(key, '')
                        turn['id'] = speaker_1 if i % 2 == 0 else speaker_2
                        if context:
                            convo['context'].append(turn)
                        else:
                            convo['dialog'].append(turn)
                json_convo = json.dumps(convo)
                f.write(json_convo + '\n')
        print(f' [ Conversations saved to file: {to_save} ]')

        # save metadata
        Metadata.save_metadata(
            to_save,
            opt,
            self_chat=self_chat,
            speaker_1=speaker_1,
            speaker_2=speaker_2,
            **kwargs,
        )
