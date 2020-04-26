#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
File adapted from
https://github.com/facebookresearch/ELI5/blob/master/data_creation/merge_support_docs.py
Modified to use data directory rather than a hard-coded processed data directory.
"""

import os
import sys
import json
from parlai.core.params import ParlaiParser
from os.path import join as pjoin 
from os.path import isdir
from data_utils import *

def setup_args():
    parser = ParlaiParser(False, False)
    parser.add_parlai_data_path()
    merge = parser.add_argument_group('Merge Support Docs')
    merge.add_argument(
        '-f', '--finalize', default=True, type=str, help='Finalize the output'
    )
    merge.add_argument(
        '-n', '--name', default='explainlikeimfive', type=str, help='Slices to merge'
    )
    return parser.parse_args()


if __name__ == '__main__':
    opt = setup_args()
    name = opt['name']
    ca = opt['finalize']
    if ca == 'finalize':
        rd_dir = pjoin(opt['datapath'], 'eli5/processed_data/collected_docs', name)
        sl_dir = pjoin(rd_dir, 'slices')
        if not isdir(sl_dir):
            os.mkdir(sl_dir)
        num_slice = 0
        docs = []
        for i in range(10):
            docs += json.load(open(pjoin(rd_dir, '%d.json' % (i,))))
            while len(docs) > 3000:
                print('writing slice', num_slice, name)
                json.dump(
                    docs[:3000], open(pjoin(sl_dir, 'slice_%d.json' % num_slice), 'w')
                )
                docs = docs[3000:]
                num_slice += 1
        if len(docs) > 0:
            json.dump(
                docs[:3000], open(pjoin(sl_dir, 'slice_%d.json' % num_slice), 'w')
            )
    else:
        d_name = pjoin(opt['datapath'], 'eli5/processed_data/collected_docs', name, ca)
        if isdir(d_name):
            merged = merge_support_docs(d_name)
        if len(merged) > 0:
            json.dump(
                merged,
                open(
                    pjoin(
                        opt['datapath'], 'eli5/processed_data/collected_docs', name, ca
                    )
                    + '.json',
                    'w',
                ),
            )
