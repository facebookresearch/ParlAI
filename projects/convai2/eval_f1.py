# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Base script for running official ConvAI2 validation eval for f1.
This uses a the version of the dataset which does not contain candidates.
Leaderboard scores will be run in the same form but on a hidden test set.
"""

from examples.eval_model import setup_args, eval_model as main_eval_model

def eval_model(parser):
    parser.set_defaults(
        task='convai2:self:no_cands',
        datatype='valid',
        hide_labels=True,
    )
    return main_eval_model(parser)


if __name__ == '__main__':
    eval_model(setup_args())
