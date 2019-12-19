#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base script for running official ConvAI2 validation eval for hits@1.

This uses a the version of the dataset which contains candidates. Leaderboard scores
will be run in the same form but on a hidden test set.
"""

from parlai.scripts.eval_model import eval_model, setup_args as base_setup_args


def setup_args(parser=None):
    parser = base_setup_args(parser)
    parser.set_defaults(
        task='convai2:self',
        datatype='valid',
        hide_labels=False,
        dict_tokenizer='split',
        metrics='hits@1',
    )
    return parser


def eval_hits(opt, print_parser):
    report = eval_model(opt, print_parser)
    print('============================')
    print('FINAL Hits@1: ' + str(report['hits@1']))
    return report


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(model='repeat_label')
    opt = parser.parse_args()
    eval_hits(opt, parser)
