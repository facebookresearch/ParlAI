#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Dialogue safety related datasets and teachers.
"""

import json
import os

from parlai.core.teachers import FixedDialogTeacher
from .build import build
from parlai.tasks.multinli.agents import convert_to_dialogData

ENTRY_FIELDS = [
    'id',
    'text',
    'labels',
    'reward',
    'label_candidates',
    'episode_done',
    'image',
]
DNLI_LABEL_DICT = {
    'positive': 'entailment',
    'negative': 'contradiction',
    'neutral': 'neutral',
}
DNLI_ANSWER_KEY = 'label'
DNLI_PREMISE_KEY = 'sentence1'
DNLI_HYPO_KEY = 'sentence2'


class DialogueNliTeacher(FixedDialogTeacher):
    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('DNLI Teacher Args')
        parser.add_argument(
            '-dfm',
            '--dialog-format',
            type='bool',
            default=False,
            help="True if one would like to convert to a dialogue format without special tokens such as 'Premise'"
            " and 'Hypothesis' (default: False).",
        )
        parser.add_argument(
            '-bcl',
            '--binary-classes',
            type='bool',
            default=False,
            help="True if label candidates are (contradiction, not_contradiction), and (entailment, contradiction, "
            "neutral) otherwise (default: False).",
        )

    def __init__(self, opt, shared=None, extras=False):
        super().__init__(opt, shared)

        # Build the data if it doesn't exist.
        build(opt)

        suffix = self.datatype
        if suffix.startswith('train'):
            suffix = 'train'
        elif suffix.startswith('test'):
            suffix = 'test'
        elif suffix.startswith('valid'):
            suffix = 'dev'

        if extras:
            datapath = os.path.join(
                opt['datapath'],
                'dialogue_nli',
                'dnli',
                'dialogue_nli_extra',
                'dialogue_nli_EXTRA_uu_' + suffix + '.jsonl',
            )
        else:
            datapath = os.path.join(
                opt['datapath'],
                'dialogue_nli',
                'dnli',
                'dialogue_nli',
                'dialogue_nli_' + suffix + '.jsonl',
            )

        self._setup_data(datapath)
        self.id = 'dnli'.upper()
        self.dialog_format = opt.get('dialog_format', False)
        self.binary_classes = opt.get('binary_classes', False)
        self.reset()

    def _setup_data(self, path):
        with open(path) as data_file:
            if 'extra' in path and 'train' in path:
                line = data_file.readline()

                # trim corrupted JSON
                line = line[: line.rfind("{")]
                line = line[: line.rfind(",")] + "]"

                self.data = json.loads(line)
            else:
                self.data = json.load(data_file)

    def num_examples(self):
        return len(self.data)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx=0):
        entry = self.data[episode_idx]
        entry['id'] = self.id
        entry['episode_done'] = True
        (
            entry['text'],
            entry['labels'],
            entry['label_candidates'],
        ) = convert_to_dialogData(
            premise_raw=entry[DNLI_PREMISE_KEY],
            hypo_raw=entry[DNLI_HYPO_KEY],
            answer_raw=DNLI_LABEL_DICT[entry[DNLI_ANSWER_KEY]],
            dialog_format=self.dialog_format,
            binary_classes=self.binary_classes,
        )
        new_entry = {k: entry[k] for k in ENTRY_FIELDS if k in entry}
        return new_entry


class ExtrasTeacher(DialogueNliTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared, extras=True)


class DefaultTeacher(DialogueNliTeacher):
    pass
