#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from parlai.utils.io import PathManager
from .build import build
import os
import json


class ELI5Teacher(FixedDialogTeacher):
    """
    ELI5 Teacher, taken from https://github.com/facebookresearch/ELI5.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        build(opt)
        self.id = 'eli5'
        self.messages = self.load_eli5(opt)
        self.reset()

    @staticmethod
    def add_cmdline_args(argparser):
        group = argparser.add_argument_group('ELI5 Knowledge arguments')
        group.add_argument(
            '--knowledge',
            type='bool',
            default=True,
            help='Whether to include supporting document knowledge',
        )

    def load_eli5(self, opt):
        """
        Load data based on data split.
        """
        dp = opt['datapath']
        dt = opt['datatype'].split(':')[0]
        eli_path = "eli5/processed_data/selected_15_1/explainlikeimfive_"
        fname = os.path.join(dp, eli_path + dt + ".json")
        if not PathManager.exists(fname):
            raise FileNotFoundError(
                f"{fname} not found. Please follow the instructions found at "
                "https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/eli5/README.md"
                " to construct the dataset."
            )
        opt['datafile'] = fname
        with PathManager.open(fname) as json_file:
            data = json.load(json_file)
        ds = []
        for d in data:
            if self.opt['knowledge']:
                text = d['document'] + "\n" + d['question']
            else:
                text = d['question']
            act = {
                'id': 'eli5',
                'text': text,
                'labels': [d['answer']],
                'episode_done': True,
            }
            ds.append(act)
        return ds

    def get(self, episode_idx, entry_idx=0):
        return self.messages[episode_idx]

    def num_examples(self):
        return len(self.messages)

    def num_episodes(self):
        return len(self.messages)


class DefaultTeacher(ELI5Teacher):
    pass
