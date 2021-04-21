#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from parlai.utils.data import DatatypeHelper
from .build import build
import os
import pandas as pd
import hashlib


class MetalWozTeacher(DialogTeacher):
    def _path(self, opt):
        fold = DatatypeHelper.fold(opt['datatype'])
        if fold == 'train' or fold == 'valid':
            folder = os.path.join(opt['datapath'], 'metalwoz', 'train')
        else:
            folder = os.path.join(opt['datapath'], 'metalwoz', 'test')
        return folder, fold

    def __init__(self, opt, shared=None):
        if shared is None:
            build(opt)
        folder, fold = self._path(opt)
        self.fold = fold
        opt['datafile'] = os.path.join(folder, fold)
        super().__init__(opt, shared)

    def _hash(self, string):
        return int(hashlib.sha1(string.encode('utf-8')).hexdigest(), 16) % 10

    def setup_data(self, datapath):
        folder, fold = os.path.split(datapath)
        with PathManager.open(os.path.join(folder, 'tasks.txt')) as taskf:
            tasks_table = pd.read_json(taskf, lines=True)

        dfolder = os.path.join(folder, 'dialogues')

        data = []

        for filename in PathManager.ls(dfolder):
            fullfn = os.path.join(dfolder, filename)
            with PathManager.open(fullfn) as dataf:
                data.append(pd.read_json(dataf, lines=True))

        data = pd.concat(data, axis=0)
        data = data.sample(frac=1.0, random_state=83741)  # metal in l33t numbers, lol
        data = data.merge(tasks_table, on='task_id')
        data['fold'] = data['domain_x'].apply(self._hash)

        for _, row in data.iterrows():
            if fold == 'valid' and row['fold'] != 9:
                continue
            if fold == 'train' and row['fold'] == 9:
                continue
            texts = [row['bot_role']] + list(row['turns'])
            prompts, labels = texts[::2], texts[1::2]
            for i, (prompt, label) in enumerate(zip(prompts, labels)):
                yield {
                    'text': prompt,
                    'label': label,
                    'bot_role': row['bot_role'],
                    'bot_prompt': row['bot_prompt'],
                    'user_role': row['user_role'],
                    'user_prompt': row['user_prompt'],
                    'utterance_id': row['id'],
                    'domain': row['domain_x'],
                    'task_id': row['task_id'],
                }, i == 0


class DefaultTeacher(MetalWozTeacher):
    pass
