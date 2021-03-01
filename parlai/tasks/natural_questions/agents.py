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
from typing import List, Optional, Tuple

import parlai.utils.logging as logging
from parlai.core.teachers import ChunkTeacher
from .build import build, DATASET_NAME_LOCAL
from .utils.text_utils import simplify_nq_example

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser


def _count_lines_in_file(fname):
    num_lines = 0
    with open(fname, 'r') as fi:
        for _ in fi:
            num_lines += 1
    return num_lines


def _context_type_key(is_html):
    return 'document_html' if is_html else 'document_text'


def _create_long_answer_from_span_html(example):
    """
    Creates a list of long answer candidates, from their spans on the document.

    This function gets the full article from the input example dictionary (using
    key 'document_html'), then iterates through the long answer spans (from
    'long_answer_candidates' key) and creates a list of slices from the article,
    using the 'start_byte' and 'end_byte' values in the list of long answer
    candidate spans.

    Returns a list of long answers. Each long answer is a substring from the
        original HTML text.

    :param example: a dict that contains one example/entry from NQ dataset.
    """
    context_text = example[_context_type_key(is_html=True)].encode()
    candidate_long_answers = []
    for long_answer_span in example['long_answer_candidates']:
        start_index = long_answer_span['start_byte']
        end_index = long_answer_span['end_byte']
        answer = context_text[start_index:end_index].decode()
        candidate_long_answers.append(answer)
    return candidate_long_answers


def _create_long_answer_from_span_text(simplified_example):
    """
    Creates a list of long answer candidates, from their spans on the document.

    This function gets the full article from the input simplified example
    dictionary (using key 'document_text'), then iterates through the long
    answer spans (from 'long_answer_candidates' key) and creates a list of
    slices from the article, using the 'start_token' and 'end_token' values in
    the list of long answer candidate spans.

    Returns a list of long answers. Each long answer is a substring from the
        simplified HTML text.

    :param simplified_example: a dict that contains one simplified example/entry
        from NQ dataset.
    """
    context_text = simplified_example[_context_type_key(is_html=False)]
    candidate_long_answers = []
    splitted_tokens = context_text.split(' ')
    for long_answer_span in simplified_example['long_answer_candidates']:
        start_index = long_answer_span['start_token']
        end_index = long_answer_span['end_token']
        answer = ' '.join(splitted_tokens[start_index:end_index])
        candidate_long_answers.append(answer)
    return candidate_long_answers


class NaturalQuestionsTeacher(ChunkTeacher):
    """
    The base teacher class for Natural Questions dataset challenge.

    This class implements the core functionalities for other teachers. The other four
    variations of teachers are made by setting two object attributes (use_html,
    use_long_answer) to either True or False.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        nq = parser.add_argument_group('Natural Questions Teacher')
        nq.add_argument(
            '--use-html',
            type='bool',
            default=False,
            help='Use HTML for the context (does nothing if `use-context` is False)',
        )
        nq.add_argument(
            '--use-long-answer', type='bool', default=False, help='Use long answers'
        )
        nq.add_argument(
            '--use-context',
            type='bool',
            default=True,
            help='Include context blurb or not',
        )

    def __init__(self, opt, shared=None):
        build(opt)
        self.use_html = opt.get('use_html', False)
        self.use_long_answer = opt.get('use_long_answer', False)
        self.use_context = opt.get('use_context', False)
        self.id = 'natural_questions'
        self.opt = copy.deepcopy(opt)
        self.dtype = self.opt['datatype'].split(':')[0]
        self.dpath = os.path.join(self.opt['datapath'], DATASET_NAME_LOCAL, self.dtype)
        self.n_samples = None
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
        elif 'valid' == self.dtype:
            return list(range(5))
        raise ValueError(f'Invalid data type: "{self.dtype}"')

    def get_num_samples(self, opt) -> Tuple[int, int]:
        if self.n_samples:
            return self.n_samples
        logging.log(f'Counting the number of samples in {self.dtype}')
        files = os.listdir(self.dpath)
        n_samples = 0
        for fname in tqdm(files):
            if fname.startswith('.'):  # some of the OS specific files
                continue
            n_samples += _count_lines_in_file(os.path.join(self.dpath, fname))
        logging.info(f'{n_samples} examples found in {self.dtype} dataset.')
        self.n_samples = (n_samples, n_samples)
        return self.n_samples

    def _get_candidate_long_answers(self, example):
        if self.use_html:
            return _create_long_answer_from_span_html(example)
        else:
            return _create_long_answer_from_span_text(example)

    def _get_short_answers(self, example):
        context = example[_context_type_key(self.use_html)]
        if self.use_html:
            offset_unit = 'byte'
            context = context.encode()
        else:
            offset_unit = 'token'
            context = context.split(' ')

        short_answers = []
        for annotation in example['annotations']:
            if 'short_answers' in annotation and annotation['short_answers']:
                for sa in annotation['short_answers']:
                    start_ind = sa[f'start_{offset_unit}']
                    end_ind = sa[f'end_{offset_unit}']
                    ans = context[start_ind:end_ind]
                    if self.use_html:
                        short_answers.append(ans.decode())
                    else:
                        short_answers.append(' '.join(ans))
            elif (
                'yes_no_answer' in annotation
                and annotation['yes_no_answer']
                and annotation['yes_no_answer'] != 'NONE'
            ):
                short_answers.append(annotation['yes_no_answer'])
        return short_answers

    def load_from_chunk(self, chunk_idx: int):
        """
        Loads from a chunk of the dataset, given the chunk index.

        Returns a list of dictionaries. Each dictionary is an example from the
            main dataset and stores the components of that examples (e.g.,
            contenxt, question, candidate answers etc.) as key-value pairs.

        :param chunk_idx: the index of the chunk dataset chunk file.
        """

        def _extract_labels_indices(example, candidate_labels):
            labels = []
            for label in example['annotations']:
                label_ind = label['long_answer']['candidate_index']
                labels.append(candidate_labels[label_ind])
            return labels

        fname = f'nq-{self.dtype}-{str(chunk_idx).zfill(2)}.jsonl'
        fpath = os.path.join(self.dpath, fname)
        output = []
        with jsonlines.open(fpath, 'r') as fi:
            for example in fi:
                example_components = dict()
                example = self._simplify(example)
                question = example['question_text']
                if self.use_context:
                    context = example[_context_type_key(self.use_html)]
                    example_components['text'] = f'{context}\n{question}?'
                else:
                    example_components['text'] = f'{question}?'

                if self.use_long_answer:
                    example_components[
                        'long_answers_candidate'
                    ] = self._get_candidate_long_answers(example)
                    example_components['long_answers'] = _extract_labels_indices(
                        example, example_components['long_answers_candidate']
                    )
                else:
                    example_components['short_answers'] = self._get_short_answers(
                        example
                    )
                output.append(example_components)
        return output

    def create_message(self, example_components, entry_idx=0):
        label_key = 'long_answers' if self.use_long_answer else 'short_answers'
        message_dict = {
            'id': self.id,
            'text': example_components['text'],
            'labels': example_components[label_key] or [''],
            'episode_done': True,
        }
        if self.use_long_answer:
            message_dict['label_candidates'] = example_components[
                'long_answers_candidate'
            ]
        return message_dict


class DefaultTeacher(NaturalQuestionsTeacher):
    pass
