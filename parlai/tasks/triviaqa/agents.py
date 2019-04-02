#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
import parlai.core.agents as core_agents
from .build import build

import copy
import json
import os


def _path(opt):
    build(opt)

    return (os.path.join(opt['datapath'], 'TriviaQA', 'qa'),
            os.path.join(opt['datapath'], 'TriviaQA', 'evidence'))


class WebTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        if not hasattr(self, 'prefix'):
            self.prefix = ''
            if opt['datatype'].startswith('train'):
                self.suffix = 'train'
            else:
                self.suffix = 'dev'

        qa_dir, self.evidence_dir = _path(opt)
        opt['datafile'] = os.path.join(
            qa_dir, self.prefix + 'web-' + self.suffix + '.json'
        )
        self.id = 'triviaqa'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            data = json.load(data_file)['Data']
        for datapoint in data:
            question = datapoint['Question']
            answers = datapoint['Answer']['Aliases']
            evidence_list = datapoint['SearchResults']

            if len(evidence_list) == 0:
                continue

            for evidence_item in evidence_list:
                evidence_file_path = os.path.join(self.evidence_dir, 'web',
                                                  evidence_item['Filename'])
                with open(evidence_file_path) as evidence_file:
                    evidence = 'Title: %s\n' % evidence_item['Title']
                    evidence += evidence_file.read()
                    yield (evidence + '\n' + question, answers), True


class VerifiedWebTeacher(WebTeacher):
    def __init__(self, opt, shared=None):
        self.prefix = 'verified-'
        self.suffix = 'dev'
        if opt['datatype'] != 'valid':
            print('WARNING: Verified teacher only provides dev data')

        opt['datafile'], self.evidence_dir = _path(opt)
        self.id = 'triviaqa'
        super().__init__(opt, shared)


class WikipediaTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        if not hasattr(self, 'prefix'):
            self.prefix = ''
            if opt['datatype'].startswith('train'):
                self.suffix = 'train'
            else:
                self.suffix = 'dev'

        qa_dir, self.evidence_dir = _path(opt)
        opt['datafile'] = os.path.join(
            qa_dir, self.prefix + 'wikipedia-' + self.suffix + '.json'
        )

        self.id = 'triviaqa'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            data = json.load(data_file)['Data']
        for datapoint in data:
            question = datapoint['Question']
            answers = datapoint['Answer']['Aliases']
            evidence_list = datapoint['EntityPages']

            if len(evidence_list) == 0:
                continue

            evidence = ''
            for evidence_item in evidence_list:
                evidence_file_path = os.path.join(self.evidence_dir,
                                                  'wikipedia',
                                                  evidence_item['Filename'])
                with open(evidence_file_path) as evidence_file:
                    evidence += 'Title: %s\n' % evidence_item['Title']
                    evidence += evidence_file.read() + '\n\n'

            yield (evidence + question, answers), True


class VerifiedWikipediaTeacher(WikipediaTeacher):
    def __init__(self, opt, shared=None):
        self.prefix = 'verified-'
        self.suffix = 'dev'
        if opt['datatype'] != 'valid':
            print('WARNING: Verified teacher only provides dev data')

        opt['datafile'], self.evidence_dir = _path(opt)
        self.id = 'triviaqa'
        super().__init__(opt, shared)


class VerifiedTeacher(core_agents.MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = 'triviaqa:VerifiedWikipedia,triviaqa:VerifiedWeb'
        super().__init__(opt, shared)


class DefaultTeacher(core_agents.MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = 'triviaqa:wikipedia,triviaqa:web'
        super().__init__(opt, shared)
