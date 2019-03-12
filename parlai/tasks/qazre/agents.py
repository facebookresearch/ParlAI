#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

from parlai.core.teachers import DialogTeacher
from .build import build
import os


class QAZRETeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.dt = opt['datatype'].split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'qazre'

        # build the data
        build(opt)

        # set the download path
        opt['datafile'] = os.path.join(opt['datapath'], 'QA-ZRE', 'relation_splits')

        super().__init__(opt, shared)

    def setup_data(self, input_path):

        print('loading: ' + input_path)

        new_episode = True

        # function to parse the episodes
        def extract_qa(qa_data):
            line_data = qa_data.split('\t')
            question_type, anon_question, deanon, context = line_data[:4]
            answer = line_data[4:]
            if answer == []:
                answer = ['No answer']
            return context + '\n' + anon_question.replace('XXX', deanon), answer

        # parse and output the episodes
        for fname in os.listdir(input_path):
            if fname.startswith(self.dt):
                with open(os.path.join(input_path, fname)) as file:
                    file_data = file.read().split('\n')[:-1]
                for line in file_data:
                    question, answer = extract_qa(line)

                    yield (question, answer, None, None), new_episode


class DefaultTeacher(QAZRETeacher):
    pass
