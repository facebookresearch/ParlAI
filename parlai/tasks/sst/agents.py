#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from .build import build
import os


class SSTTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.dt = opt['datatype'].split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'sst'

        self.SST_LABELS = ['negative', 'positive']

        opt['datafile'] = self._path(opt)

        super().__init__(opt, shared)

    def _path(self, opt):
        build(opt)

        dt = opt['datatype'].split(':')[0]
        # Using matched set as valid and mismatched set as test
        if dt == 'valid':
            dt = 'dev'
        fname = dt + '_binary_sent.csv'
        path = os.path.join(opt['datapath'], 'SST', fname)

        return path

    def setup_data(self, path):
        print('loading: ' + path)

        with PathManager.open(path) as data_file:
            self.all_lines = [
                l.strip().split(',', 1) for l in data_file.read().split("\n")[1:-1]
            ]

            self.labels = [self.SST_LABELS[int(x[0])] for x in self.all_lines]
            self.contexts = [x[1] for x in self.all_lines]

        new_episode = True
        # define standard question, since it doesn't change for this task

        # every episode consists of only one query in this task
        self.question = 'Is this sentence positive or negative?'

        # define iterator over all queries

        for i in range(len(self.contexts)):
            if self.labels[i]:
                yield (
                    self.contexts[i] + '\n' + self.question,
                    [self.labels[i]],
                    None,
                    None,
                ), new_episode

    def label_candidates(self):
        return self.SST_LABELS


class DefaultTeacher(SSTTeacher):
    pass
