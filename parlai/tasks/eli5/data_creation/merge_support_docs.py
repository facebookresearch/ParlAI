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
import json
from parlai.core.params import ParlaiParser
from os.path import join as pjoin
from os.path import isdir, isfile
from glob import glob
from data_utils import merge_support_docs


def setup_args():
    parser = ParlaiParser(False, False)
    parser.add_parlai_data_path()
    merge = parser.add_argument_group('Merge Support Docs')
    merge.add_argument(
        '-c',
        '--c',
        default=True,
        type=str,
        help='Finalize the output (or slice number to merge)',
    )
    merge.add_argument(
        '-n', '--name', default='explainlikeimfive', type=str, help='Slices to merge'
    )
    return parser.parse_args()


# merge CC docs that weren't created for eli5
def merge_non_eli_docs(doc_name):
    docs = []
    merged = {}
    for f_name in glob(pjoin(doc_name, '*.json')):
        docs += json.load(open(f_name))
    if not docs or len(docs[0]) < 3:
        for i, (num, article) in enumerate(docs):
            merged[i] = merged.get(i, [''] * 100)
            merged[i][num] = article
    else:
        return None
    for eli_k, articles in merged.items():
        merged[eli_k] = [art for art in articles if art != '']
        merged[eli_k] = [
            x
            for i, x in enumerate(merged[eli_k])
            if (x['url'] not in [y['url'] for y in merged[eli_k][:i]])
        ]
    return list(merged.items())


if __name__ == '__main__':
    opt = setup_args()
    name = opt['name']
    ca = opt['c']
    if ca == 'finalize':
        rd_dir = pjoin(opt['datapath'], 'eli5/processed_data/collected_docs', name)
        sl_dir = pjoin(rd_dir, 'slices')
        if not isdir(sl_dir):
            os.mkdir(sl_dir)
        num_slice = 0
        docs = []
        for i in range(10):
            if not isfile(pjoin(rd_dir, '%d.json' % (i,))):
                continue
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
            # merge docs in single slice (0-9) directory
            non_eli_merged = merge_non_eli_docs(d_name)
            if non_eli_merged is None:
                merged = merge_support_docs(d_name)
            else:
                merged = non_eli_merged
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
