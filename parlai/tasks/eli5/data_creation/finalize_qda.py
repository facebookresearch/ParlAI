#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

from glob import glob
from os.path import join as pjoin
from parlai.core.params import ParlaiParser

"""
Adapted from https://github.com/facebookresearch/ELI5/blob/main/data_creation/finalize_qda.py
to use data directory rather than a hard-coded directory
"""


def setup_args():
    parser = ParlaiParser(False, False)
    parser.add_parlai_data_path()
    finalize = parser.add_argument_group('Gather into train, valid and tests')
    finalize.add_argument(
        '-ns',
        '--num_selected',
        default=15,
        type=int,
        metavar='N',
        help='number of selected passages',
    )
    finalize.add_argument(
        '-nc',
        '--num_context',
        default=1,
        type=int,
        metavar='N',
        help='number of sentences per passage',
    )
    finalize.add_argument(
        '-sr_l',
        '--subreddit_list',
        default='["explainlikeimfive"]',
        type=str,
        help='subreddit name',
    )
    return parser.parse_args()


def main():
    opt = setup_args()
    n_sel = opt['num_selected']
    n_cont = opt['num_context']
    dpath = opt['datapath']
    for name in json.loads(opt['subreddit_list']):
        data_split = json.load(open('pre_computed/%s_split_keys.json' % (name,)))
        qda_list = []
        slice_json = pjoin(
            dpath,
            'eli5/processed_data/selected_%d_%d/%s/selected_slice_*.json'
            % (n_sel, n_cont, name),
        )
        for f_name in glob(slice_json):
            qda_list += json.load(open(f_name))
        qda_dict = dict([(dct['id'], dct) for dct in qda_list])
        for spl in ['train', 'valid', 'test']:
            split_list = [qda_dict[k] for k in data_split[spl] if k in qda_dict]
            selected_json = pjoin(
                dpath,
                'eli5/processed_data/selected_%d_%d/%s_%s.json'
                % (n_sel, n_cont, name, spl),
            )
            json.dump(split_list, open(selected_json, 'w'))


if __name__ == '__main__':
    main()
