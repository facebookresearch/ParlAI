#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import json
import logging
import os

from parlai.core.teachers import DialogTeacher
from parlai.core.opt import Opt
from parlai.utils.data import DatatypeHelper


from .build import build, DATASET_NAME


def get_dtype(opt):
    return DatatypeHelper.fold(opt.get('datatype', 'train'))


def _path(opt):
    build(opt)
    dpath = os.path.join(opt['datapath'], DATASET_NAME)
    dtype = get_dtype(opt)
    if dtype == 'valid':
        logging.warning(
            'This data set does not have valid split. Using `test` instead.'
        )
        dtype = 'test'
    return os.path.join(dpath, f'{dtype}.jsonl')


class BaseJerichoWorldTeacher(DialogTeacher):
    def __init__(self, opt: Opt, shared=None):
        opt = deepcopy(opt)
        opt['datafile'] = _path(opt)
        self.datatype = get_dtype(opt)
        self.id = 'JerichoWorldtBase'
        super().__init__(opt, shared=shared)

    def setup_data(self, datafile: str):
        print(datafile)
        with open(datafile) as df:
            data = json.load(df)
