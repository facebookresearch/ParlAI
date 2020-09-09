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
from tqdm import tqdm
from typing import List, Tuple

import parlai.utils.logging as logging
from parlai.core.teachers import ChunkTeacher
from .build import build, DATASET_NAME_LOCAL
from .text_utils import simplify_nq_example


def _count_lines_in_file(fname):
    num_lines = 0
    with open(fname, 'r') as fi:
        for line in fi:
            num_lines += 1
    return num_lines


def _html_context_key(is_html):
    return 'document_html' if is_html else 'document_text'


def _create_long_answer_from_span(example, html_example):
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
    def _token_or_byte():
        return 'byte' if html_example else 'token'

    context_text = example[_html_context_key(html_example)]
    candidate_long_answers = []
    for long_asnwer_span in example['long_answer_candidates']:
        if not long_asnwer_span['top_level']:
            # not including answers contained in other ones.
            continue
        start_index_byte = long_asnwer_span[f'start_{_token_or_byte()}'] - 1
        end_index_byte = long_asnwer_span[f'end_{_token_or_byte()}'] - 1
        candidate_long_answers.append(
            context_text[start_index_byte:end_index_byte])
    return candidate_long_answers


class LongAnswerTeacher(ChunkTeacher):
    """
    Dialog Teacher for long answer format in Natural Questions

    NOTE: This implementation of teacher is inefficient, to the extent of being
    unpractical. This is due to the size of the dataset. It may only be used
    with the toy (e'g', provided sample) datasets.
    """
    def __init__(self, opt, shared=None):
        self.use_html = opt.get('use_html', False)
        build(opt)
        self.id = 'natural_questions'
        self.opt = copy.deepcopy(opt)
        self.dtype = self.opt['datatype'].split(':')[0]
        self.dpath = os.path.join(self.opt['datapath'], DATASET_NAME_LOCAL, self.dtype)
        super().__init__(self.opt, shared)

    def _simplify(self, example):
        if self.use_html:
            return example
        return simplify_nq_example(example)

    def _get_data_folder(self):
        return self.dpath

    def get_fold_chunks(self, opt) -> List[int]:
        if 'train' == self.dtype:
            return list(range(50))
        elif 'dev' == self.dtype:
            return list(range(5))
        raise ValueError(f'Invalid data type: "{self.dtype}"')

    def get_num_samples(self, opt) -> Tuple[int, int]:
        logging.log(f'Counting the number of samples in {self.dtype}')
        files = os.listdir(self.dpath)
        n_samples = 0
        for fname in tqdm(files):
            if fname.startswith('.'):  # some of the OS specific files
                continue
            n_samples += _count_lines_in_file(os.path.join(self.dpath, fname))
        logging.info(f'{n_samples} examples found in {self.dtype} dataset.')
        return (n_samples, n_samples)

    def load_from_chunk(self, chunk_idx: int):
        fname = f'nq-{self.dtype}-{str(chunk_idx).zfill(2)}.jsonl'
        fpath = os.path.join(self.dpath, fname)
        output = []
        with jsonlines.open(fpath, 'r') as fi:
            for example in fi:
                example = self._simplify(example)
                context = example[_html_context_key(self.use_html)]
                question = example['question_text']
                answers = _create_long_answer_from_span(example, self.use_html)
                output.append((f'{context}\n{question}?', answers))
        return output

    def create_message(self, sample_item, entry_idx=0):
        text, labels = sample_item
        return {'id': self.id, 'text': text, 'labels': labels, 'episode_done': True}

class DefaultTeacher(LongAnswerTeacher):
    pass
