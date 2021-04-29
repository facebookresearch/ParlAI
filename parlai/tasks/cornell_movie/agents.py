#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from .build import build
from parlai.utils.data import DatatypeHelper

import copy
import os
import codecs


def _path(opt, *additions):
    return os.path.join(opt['datapath'], 'CornellMovie', *additions)


class DefaultTeacher(DialogTeacher):
    DOUBLE = False

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        self.fold = DatatypeHelper.fold(opt['datatype'])
        build(opt)
        opt['datafile'] = _path(opt, self.fold + '.txt')
        super().__init__(opt, shared)

    def setup_data(self, datafile):
        lines_file = _path(self.opt, 'cornell movie-dialogs corpus', 'movie_lines.txt')
        convo_file = _path(
            self.opt, 'cornell movie-dialogs corpus', 'movie_conversations.txt'
        )

        lines = {}

        codecs.register_error('strict', codecs.ignore_errors)
        with codecs.open(lines_file, 'r') as f:
            for line in f:
                l = line.split(' +++$+++ ')
                lines[l[0]] = ' '.join(l[4:]).strip('\n').replace('\t', ' ')

        cnt = 0
        with codecs.open(convo_file, 'r') as f:
            for cnt, line in enumerate(f, 1):
                l = line.split(' ')
                convo = ' '.join(l[6:]).strip('\n').strip('[').strip(']')
                c = convo.replace("'", '').replace(' ', '').split(',')

                texts = [lines[l] for l in c]

                if (cnt % 10 == 0) and self.fold != 'test':
                    continue
                elif (cnt % 10 == 1) and self.fold != 'valid':
                    continue
                elif (cnt % 10 > 1) and self.fold != 'train':
                    continue

                for i, (prompt, response) in enumerate(zip(texts[::2], texts[1::2])):
                    yield {'text': prompt, 'label': response}, i == 0

                if self.DOUBLE:
                    for i, (prompt, response) in enumerate(
                        zip(texts[1::2], texts[2::2])
                    ):
                        yield {'text': prompt, 'label': response}, i == 0


class DoubleTeacher(DefaultTeacher):
    """
    This version creates text-label pairs from the perspective of both speakers.
    """

    DOUBLE = True
