#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file contains teacher agents for Natural Question dataset.
# General infomration about the dataset: https://ai.google.com/research/NaturalQuestions
# Details about its content: https://github.com/google-research-datasets/natural-questions

import copy
import os
import jsonlines

import parlai.utils.logging as logging
from parlai.core.teachers import DialogTeacher
from .build import build, DATASET_NAME_LOCAL

def _add_datafiles_path_to_opt(opt):
    if opt['datatype'].startswith('train'):
        datadir = 'train'
    else:
        datadir = 'dev'
    opt['datafile'] = os.path.join(
        opt['datapath'], DATASET_NAME_LOCAL, datadir)

def _create_long_answer_from_span(example):
    """
    Creates a list of long answer candidates, from their spans and the document.

    This functin gets the full article from the input example dictionary (using
    key 'document_html'), then iterates through the long answer spans (from
    'long_answer_candidates' key) and creates a list of slices from the article,
    using the 'start_byte' and 'end_byte' values in the list of long answer
    candidate spans.

    Returns a list of long answers. Each long answer is an str (html text).

    :param example: a dict that contain one example/entry from NQ dataset.
    """
    context_text = example['document_html']
    candidate_long_answers = []
    for long_asnwer_span in example['long_answer_candidates']:
        if not long_asnwer_span['top_level']:
            # not including answers contained in other ones.
            continue
        start_index_byte = long_asnwer_span['start_byte'] - 1
        end_index_byte = long_asnwer_span['end_byte'] - 1
        candidate_long_answers.append(
            context_text[start_index_byte:end_index_byte])
    return candidate_long_answers


class LongAnswerTeacher(DialogTeacher):
    """
    Dialog Teacher for long answer format in Natural Questions

    NOTE: This implementation of teacher is inefficient, to the extent of being
    unpractical. This is due to the size of the dataset. It may only be used
    with the toy (e'g', provided sample) datasets.
    """
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        self.use_html = opt.get('use_html', False)
        build(opt)
        self.id = 'natural_questions'
        opt = copy.deepcopy(opt)
        _add_datafiles_path_to_opt(opt)
        super().__init__(opt, shared)

    def _transform_html(self, html_content):
        if self.use_html:
            return html_content
        # TODO(Mojtaba Komeili): implement tranformation later
        return html_content  #  implement the transformation to plain text

    def setup_data(self, path):
        logging.info(f'reading input from files in {path}')
        input_files = os.listdir(path)
        for fname in input_files:
            fpath = os.path.join(path, fname)
            logging.info(f'reading from {fname}')
            with jsonlines.open(fpath, 'r') as fi:
                for example in fi:
                    context = self._transform_html(example['document_html'])
                    question = example['question_text']
                    answers = _create_long_answer_from_span(example)
                    answers = tuple(self._transform_html(a) for a in answers)
                    yield (f'{context}\n{question}?', answers), True


class DefaultTeacher(LongAnswerTeacher):
    pass
