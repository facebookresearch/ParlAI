# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.dialog_teacher import DialogTeacher
from .build import build

import json
import os
import random

def _path(opt):
    build(opt)

    return (os.path.join(opt['datapath'], 'TriviaQA', 'qa'),
            os.path.join(opt['datapath'], 'TriviaQA', 'evidence'))

class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        if opt['datatype'].startswith('train'):
            self.suffix = 'train'
        else:
            self.suffix = 'dev'

        opt['datafile'], self.evidence_dir = _path(opt)
        self.id = 'triviaqa'
        super().__init__(opt, shared)

    def setup_data(self, path):
        datasets = ['web', 'wikipedia']

        for dataset in datasets:
            dataset_path = os.path.join(path,
                                        dataset + '-' + self.suffix + '.json')
            print('loading: ' + dataset_path)
            with open(dataset_path) as data_file:
                self.triviaqa = json.load(data_file)['Data']
            for datapoint in self.triviaqa:
                question = datapoint['Question']
                answers = datapoint['Answer']['Aliases']
                if dataset == 'web':
                    evidence_list = datapoint['SearchResults']
                else:
                    evidence_list = datapoint['EntityPages']

                if len(evidence_list) == 0:
                    continue

                evidence_num = random.randrange(len(evidence_list))
                evidence_filename = evidence_list[evidence_num]['Filename']
                evidence_file_path = os.path.join(self.evidence_dir, dataset,
                                                  evidence_filename)
                with open(evidence_file_path) as evidence_file:
                    evidence = evidence_file.read()
                    yield (evidence + '\n' + question, answers), True
