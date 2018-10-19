#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import importlib

decanlp_tasks = ['squad', 'iwslt14', 'cnn_dm', 'multinli',
                 'sst', 'qasrl', 'qazre', 'woz', 'wikisql', 'mwsc']


def build(opt):
    dpath = os.path.join(opt['datapath'], 'decanlp')
    version = 'None'

    if not build_data.built(dpath, version_string=version):
        print('Building DecaNLP Tasks...')
        build_tasks = [importlib.__import__('parlai.tasks.{}.build'.format(task),
                                            fromlist=['build'])
                       for task in decanlp_tasks]
        for task in build_tasks:
            task.build(opt)
        build_data.make_dir(dpath)
        build_data.mark_done(dpath, version_string=version)
