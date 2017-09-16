# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import json
import logging
import os

from parlai.core.dialog_teacher import DialogTeacher


class DefaultTeacher(DialogTeacher):
    """This version of Zreddit inherits from the core Dialog Teacher, which just
    requires it to define an iterator over its data `setup_data` in order to
    inherit basic metrics, a default `act` function, and enables
    Hogwild training with shared memory with no extra work.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        if opt['datatype'].startswith('train'):
            suffix = 'train'
        else:
            suffix = 'dev'
        opt['datafile'] = os.path.join(opt['datapath'], 'zreddit')
        self.id = 'zreddit'
        super().__init__(opt, shared)

    def setup_data(self, path):
        # print('loading: ' + path)
        with open(path) as data_file:
            for line in data_file:
                try:
                    # if path == '/Users/ylifb/ParlAI/data/zreddit/zelda.json':
                    #     print(line)
                    line_json = json.loads(line)
                    yield (line_json['text'], ), True
                except Exception:
                    logging.warning('data skipped: unable to parse from ' + path + ' : ' + line)
