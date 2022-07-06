#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from .build import build

import os
import copy
import csv
import glob


def _path(opt):
    build(opt)

    dt = opt['datatype'].split(':')[0]

    if not (dt == 'train' or dt == 'valid' or dt == 'test'):
        raise RuntimeError('Not valid datatype.')

    suffix = dt

    data_path = os.path.join(opt['datapath'], 'NarrativeQA', 'narrative_qa', suffix)

    return data_path


class SummariesTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        data_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'NarrativeQA'

        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading data from: ' + path)

        qa_path = os.path.join(path, 'qaps.csv')
        summaries_path = os.path.join(path, 'summaries.csv')

        qa_pairs = dict()

        with PathManager.open(qa_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['document_id'] not in qa_pairs:
                    qa_pairs[row['document_id']] = []
                qa_pairs[row['document_id']].append(row)

        with PathManager.open(summaries_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                info = 'Summary:  %s' % row['summary_tokenized']

                for i, qa in enumerate(qa_pairs[row['document_id']]):
                    question = qa['question_tokenized']
                    answer1 = qa['answer1_tokenized']
                    answer2 = qa['answer2_tokenized']

                    if i == 0:
                        # Prepend start info in first question
                        yield (info + '\n' + question, [answer1, answer2]), True
                    else:
                        yield (question, [answer1, answer2]), False


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        data_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'NarrativeQA'

        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading data from: ' + path)

        qa_path = os.path.join(path, 'qaps.csv')
        documents_path = os.path.join(path, 'documents.csv')

        stories_base_path = os.path.join(path, '..', 'stories')
        qa_pairs = dict()

        print(
            "%s stories found."
            % len(glob.glob(os.path.join(stories_base_path, "*.content")))
        )

        with PathManager.open(qa_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['document_id'] not in qa_pairs:
                    qa_pairs[row['document_id']] = []
                qa_pairs[row['document_id']].append(row)

        with PathManager.open(documents_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                story_path = os.path.join(
                    stories_base_path, row['document_id'] + '.content'
                )

                if not os.path.exists(story_path):
                    continue

                story = None
                with PathManager.open(
                    story_path, 'r', encoding='utf-8', errors='ignore'
                ) as f:
                    story = f.read().strip()

                info = 'Title:  %s' % row['wiki_title']
                info += '\nKind: %s' % row['kind']
                info += '\nStory url: %s' % row['story_url']
                info += '\nStory start: %s' % row['story_start']
                info += '\nStory end: %s' % row['story_end']
                info += '\nStory: %s' % story

                for i, qa in enumerate(qa_pairs[row['document_id']]):
                    question = qa['question_tokenized']
                    answer1 = qa['answer1_tokenized']
                    answer2 = qa['answer2_tokenized']

                    if i == 0:
                        # Prepend start info in first question
                        yield (info + '\n' + question, [answer1, answer2]), True
                    else:
                        yield (question, [answer1, answer2]), False
