# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Base script for running official ConvAI2 validation eval for f1.
This uses a the version of the dataset which does not contain candidates.
Leaderboard scores will be run in the same form but on a hidden test set.
"""

from parlai.scripts.eval_model import eval_model, setup_args as base_setup_args

def setup_args(parser=None):
    parser = base_setup_args(parser)
    parser.set_defaults(
        task='twitter',
        datatype='valid',
    )
    return parser

def eval_f1(opt, print_parser=None):
    report = eval_model(opt, print_parser=print_parser)
    print("============================")
    print("FINAL F1: " + str(report['f1']))

if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    report = eval_f1(opt, print_parser=parser)
