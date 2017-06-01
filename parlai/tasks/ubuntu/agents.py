# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.dialog_teacher import DialogTeacher
from .build import build

import csv
import random
import os


class DefaultTeacher(DialogTeacher):
    """This teacher inherits from the core Dialog Teacher, which just
    requires it to define an iterator over its data `setup_data` in order to
    inherit basic metrics, a default `act` function, and enables
    Hogwild training with shared memory with no extra work.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        opt['datafile'] = os.path.join(opt['datapath'], 'Ubuntu',
                                       opt['datatype'].split(':')[0] + '.csv')
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path, 'r', newline='') as read:
            csv_read = csv.reader(read)
            next(csv_read)  # eat header

            for line in csv_read:
                fields = [
                    s.replace('__eou__', '.').replace('__eot__', '\n').strip()
                    for s in line
                ]
                context = fields[0]
                response = fields[1]
                cands = None
                if len(fields) > 3:
                    cands = [fields[i] for i in range(2, len(fields))]
                    cands.append(response)
                    random.shuffle(cands)
                yield (context, [response], None, cands), True
