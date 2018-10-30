#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Download and build the data if it does not exist.

from parlai.core.teachers import DialogTeacher
from .build import build
import os


class SSTTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.dt = opt['datatype'].split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'sst'

        self.SST_LABELS = ['negative', 'positive']

        self.label_path, opt['datafile'] = self._path(opt)

        super().__init__(opt, shared)

    def _path(self, opt):
        build(opt)
        labels_path = os.path.join(opt['datapath'], 'SST', 'stanfordSentimentTreebank',
                                   'sentiment_labels.txt')
        data_path = os.path.join(opt['datapath'], 'SST', 'stanfordSentimentTreebank',
                                 'dictionary.txt')
        return labels_path, data_path

    def setup_data(self, path):
        print('loading: ' + path)

        def to_binary_label(label):
            sentiment_val = float(label)
            if sentiment_val <= 0.333:
                return self.SST_LABELS[0]
            elif sentiment_val > 0.666:
                return self.SST_LABELS[1]
            return False

        with open(path) as data_file:
            self.contexts = sorted(
                [data.split("|") for data in data_file.read().split("\n")[1:-1]],
                key=lambda x: int(x[1]))
            self.contexts = [x[0] for x in self.contexts]

        with open(self.label_path) as labels_file:
            self.labels = [to_binary_label(label.split("|")[1]) for label in
                           labels_file.read().split("\n")[1:-1]]

        new_episode = True
        # define standard question, since it doesn't change for this task

        # every episode consists of only one query in this task
        self.question = 'Is this sentence positive or negative?'

        # define iterator over all queries

        for i in range(len(self.contexts)):
            if self.labels[i]:
                yield (self.contexts[i] + '\n' + self.question, [self.labels[i]], None,
                       None), new_episode

    def label_candidates(self):
        return self.SST_LABELS


class DefaultTeacher(SSTTeacher):
    pass
