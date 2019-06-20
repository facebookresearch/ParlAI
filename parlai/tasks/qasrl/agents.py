#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

from parlai.core.teachers import DialogTeacher
from .build import build
import os
import copy

FILE_START = 'wiki1.'
FILE_END = '.qa'


def _path(opt):
    build(opt)

    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        suffix = 'train'
    # Using matched set as valid and mismatched set as test
    elif dt == 'valid':
        suffix = 'dev'
    elif dt == 'test':
        suffix = 'test'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], 'QA-SRL', FILE_START + suffix + FILE_END)
    return data_path


class QASRLTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        data_path = _path(opt)

        opt['datafile'] = data_path

        # store identifier for the teacher in the dialog
        self.id = 'qasrl'

        build(opt)

        super().__init__(opt, shared)

    def setup_data(self, input_path):

        print('loading: ' + input_path)
        file_path = os.path.join(input_path)

        new_episode = True

        def convert_to_qa(input_data):
            lines = input_data.split('\n')
            context = lines[1]
            predicate_count = int(lines[0].split('\t')[-1])
            unparsed_qa = lines[2:]

            def parse_qa(qa_line):
                qa_split = qa_line.split('\t?\t')
                question = (
                    context
                    + '\n'
                    + qa_split[0].replace('\t_', '').replace('\t', ' ')
                    + '?'
                )
                answers = qa_split[1].split(' ### ')
                return [question, answers]

            qa_pairs = []
            counter = 0
            for _i in range(predicate_count):
                question_count = int(unparsed_qa[counter].split('\t')[-1])
                counter += 1
                for _j in range(question_count):
                    qa_pairs.append(parse_qa(unparsed_qa[counter]))
                    counter += 1
            return qa_pairs

        with open(file_path) as file:
            # split the data by sentences
            file_data = file.read().split('\n\n')[:-1]
        for data in file_data:
            for qa in convert_to_qa(data):
                question = qa[0]
                answers = qa[1]
                yield (question, answers, None, None), new_episode


class DefaultTeacher(QASRLTeacher):
    pass
