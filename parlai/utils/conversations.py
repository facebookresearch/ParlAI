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
import itertools

from parlai.utils.io import PathManager
from parlai.core.metrics import dict_report
from parlai.utils.misc import AttrDict
import parlai.utils.logging as logging

BAR = '=' * 60
SMALL_BAR = '-' * 60


class Metadata:
    """
    Utility class for conversation metadata.

    Metadata should be saved at ``<datapath>.metadata``.
    """

    def __init__(self, datapath):
        self._load(datapath)

    def _load(self, datapath):
        self.metadata_path = self._get_path(datapath)
        if not PathManager.exists(self.metadata_path):
            raise RuntimeError(
                f'Metadata at path {self.metadata_path} not found. '
                'Double check your path.'
            )

        with PathManager.open(self.metadata_path, 'rb') as f:
            metadata = json.load(f)

        self.datetime = metadata['date']
        self.opt = metadata['opt']
        self.self_chat = metadata['self_chat']
        self.speakers = metadata['speakers']
        self.version_num = metadata['version']
        self.extra_data = {}
        for k, v in metadata.items():
            if k not in ['date', 'opt', 'speakers', 'self_chat', 'version']:
                self.extra_data[k] = v

    def read(self):
        """
        Read the relevant metadata.
        """
        string = f'Metadata version {self.version_num}\n'
        string += f'Saved at: {self.datetime}\n'
        string += f'Self chat: {self.self_chat}\n'
        string += f'Speakers: {self.speakers}\n'
        string += 'Opt:\n'
        for k, v in self.opt.items():
            string += f'\t{k}: {v}\n'
        for k, v in self.extra_data.items():
            string += f'{k}: {v}\n'

        return string

    @staticmethod
    def _get_path(datapath):
        fle, _ = os.path.splitext(datapath)
        return fle + '.metadata'

    @staticmethod
    def version():
        return '0.1'

    @classmethod
    def save_metadata(cls, datapath, opt, self_chat=False, speakers=None, **kwargs):
        """
        Dump conversation metadata to file.
        """
        metadata = {}
        metadata['date'] = str(datetime.datetime.now())
        metadata['opt'] = opt
        metadata['self_chat'] = self_chat
        metadata['speakers'] = speakers
        metadata['version'] = cls.version()

        for k, v in kwargs.items():
            metadata[k] = v

        metadata_path = cls._get_path(datapath)
        logging.info(f'Writing metadata to file {metadata_path}')
        with PathManager.open(metadata_path, 'w') as f:
            f.write(json.dumps(metadata))


class Turn(AttrDict):
    """
    Utility class for a dialog turn.
    """

    def __init__(self, id=None, text=None, **kwargs):
        super().__init__(self, id=id, text=text, **kwargs)


class Conversation:
    """
    Utility class for iterating through a single episode.

    Used in the context of the Conversations class.
    """

    def __init__(self, episode):
        self.episode = episode
        self.context = episode.get('context')
        self.metadata_path = episode.get('metadata_path')
        self.turns = self._build_turns(episode)

    def _build_turns(self, episode):
        turns = []
        for act_pair in episode['dialog']:
            for act in act_pair:
                turns.append(Turn(**act))
        return turns

    def __str__(self):
        string = BAR + '\n'
        high_level = [k for k in self.episode.keys() if k != 'dialog']
        if high_level:
            for key in high_level:
                string += f'{key}: {self.episode[key]}\n'
            string += SMALL_BAR + '\n'

        for turn in self.turns:
            string += f'{turn.id}: {turn.text}\n'

        string += BAR + '\n'
        return string

    def __len__(self):
        return len(self.turns)

    def __getitem__(self, index):
        return self.turns[index]

    def __iter__(self):
        self.iterator_idx = 0
        return self

    def __next__(self):
        """
        Return the next conversation.
        """
        if self.iterator_idx >= len(self.turns):
            raise StopIteration

        conv = self.turns[self.iterator_idx]
        self.iterator_idx += 1

        return conv


class Conversations:
    """
    Utility class for reading and writing from ParlAI Conversations format.

    Conversations should be saved in JSONL format, where each line is
    a JSON of the following form:

    WARNING: The data below must be on ONE LINE per dialogue
    in a conversation file or it will not load!!

    .. code-block:

        {
            'possible_conversation_level_info': True,
            'dialog':
                [   [
                        {
                            'id': 'speaker_1',
                            'text': <first utterance>,
                        },
                        {
                            'id': 'speaker_2',
                            'text': <second utterance>,
                        },
                        ...
                    ],
                    ...
                ]
            ...
        }
    """

    def __init__(self, datapath):
        self._datapath = datapath
        self.metadata = self._load_metadata(datapath)

    def __len__(self):
        return sum(1 for _ in self._load_raw(self._datapath))

    def _load_raw(self, datapath):
        """
        Load the data as a raw, unparsed file.

        Useful for fast IO stuff like random indexing.
        """
        if not PathManager.exists(datapath):
            raise RuntimeError(
                f'Conversations at path {datapath} not found. '
                'Double check your path.'
            )

        with PathManager.open(datapath, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                yield line

    def _parse(self, line):
        return Conversation(json.loads(line))

    def _load_conversations(self, datapath):
        return (self._parse(line) for line in self._load_raw(datapath))

    def _load_metadata(self, datapath):
        """
        Load metadata.

        Metadata should be saved at <identifier>.metadata
        Metadata should be of the following format:
        {
            'date': <date collected>,
            'opt': <opt used to collect the data>,
            'speakers': <identity of speakers>,
            ...
            Other arguments.
        }
        """
        try:
            metadata = Metadata(datapath)
            return metadata
        except RuntimeError:
            logging.debug('Metadata does not exist. Please double check your datapath.')
            return None

    def read_metadata(self):
        if self.metadata is not None:
            logging.info(self.metadata)
        else:
            logging.warning('No metadata available.')

    def __getitem__(self, index):
        raw = self._load_raw(self._datapath)
        item = list(itertools.islice(raw, index, index + 1))[0]
        return self._parse(item)

    def __iter__(self):
        return self._load_conversations(self._datapath)

    @staticmethod
    def _get_path(datapath):
        fle, _ = os.path.splitext(datapath)
        return fle + '.jsonl'

    @staticmethod
    def _check_parent_dir_exits(datapath):
        parent_dir = os.path.dirname(datapath)
        if not parent_dir or PathManager.exists(parent_dir):
            return
        logging.info(f'Parent directory ({parent_dir}) did not exist and was created.')
        PathManager.mkdirs(parent_dir)

    @classmethod
    def save_conversations(
        cls,
        act_list,
        datapath,
        opt,
        save_keys='all',
        context_ids='context',
        self_chat=False,
        **kwargs,
    ):
        """
        Write Conversations to file from an act list.

        Conversations assume the act list is of the following form: a list of episodes,
        each of which is comprised of a list of act pairs (i.e. a list dictionaries
        returned from one parley)
        """
        cls._check_parent_dir_exits(datapath)
        to_save = cls._get_path(datapath)

        context_ids = context_ids.strip().split(',')
        # save conversations
        speakers = []
        with PathManager.open(to_save, 'w') as f:
            for ep in act_list:
                if not ep:
                    continue
                convo = {
                    'dialog': [],
                    'context': [],
                    'metadata_path': Metadata._get_path(to_save),
                }
                for act_pair in ep:
                    new_pair = []
                    for ex in act_pair:
                        ex_id = ex.get('id')
                        if ex_id in context_ids:
                            context = True
                        else:
                            context = False
                            if ex_id not in speakers:
                                speakers.append(ex_id)

                        # set turn
                        turn = {}
                        if save_keys != 'all':
                            save_keys_lst = save_keys.split(',')
                        else:
                            save_keys_lst = ex.keys()
                        for key in save_keys_lst:
                            turn[key] = ex.get(key, '')
                            if key == 'metrics':
                                turn[key] = dict_report(turn[key])
                        turn['id'] = ex_id
                        if not context:
                            new_pair.append(turn)
                        else:
                            convo['context'].append(turn)
                    if new_pair:
                        convo['dialog'].append(new_pair)
                json_convo = json.dumps(convo, default=lambda v: '<not serializable>')
                f.write(json_convo + '\n')
        logging.info(f'Conversations saved to file: {to_save}')

        # save metadata
        Metadata.save_metadata(
            to_save, opt, self_chat=self_chat, speakers=speakers, **kwargs
        )
