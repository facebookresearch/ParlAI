#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from .build import build

import xml.etree.ElementTree as ET
import os
import copy


COPA = "COPA"
COPA_RESOURCES_FOLDER_NAME = 'COPA-resources'
COPA_DATASETS_FOLDER_NAME = 'datasets'
COPA_DATASET_PREFIX = 'copa-'
COPA_CAUSE_SUFFIX = "What was the CAUSE for this?"
COPA_RESULT_SUFFIX = "What happened as a RESULT?"


def _path(opt):
    build(opt)

    dt = opt['datatype'].split(':')[0]

    if dt == 'train' or dt == 'valid':
        suffix = 'dev'
    elif dt == 'test':
        suffix = 'test'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(
        opt['datapath'],
        COPA,
        COPA_RESOURCES_FOLDER_NAME,
        COPA_DATASETS_FOLDER_NAME,
        COPA_DATASET_PREFIX + suffix + '.xml',
    )

    return data_path


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        data_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'COPA'

        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)

        tree = ET.parse(path)
        root = tree.getroot()

        for child in root:
            asks_for = child.attrib['asks-for']
            answer = child.attrib['most-plausible-alternative']

            premise = child[0].text
            alternative_one = child[1].text
            alternative_two = child[2].text

            if asks_for == "cause":
                premise += " " + COPA_CAUSE_SUFFIX
            else:
                premise += " " + COPA_RESULT_SUFFIX

            if answer == "1":
                answer = [alternative_one]
            else:
                answer = [alternative_two]
            answer_options = [alternative_one, alternative_two]

            yield (premise, answer, None, answer_options), True
