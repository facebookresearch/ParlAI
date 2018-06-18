# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


from parlai.core.agents import Agent

from parlai.scripts.eval_ppl import eval_ppl as run_eval_ppl, setup_args as setup_ppl_args
from projects.twitter.build_dict import build_dict

import math


def setup_args(parser=None):
    parser = setup_ppl_args(parser)
    parser.set_defaults(
        task='twitter',
        datatype='valid',
        metrics='ppl',
    )
    return parser


def eval_ppl(opt):
    return run_eval_ppl(opt, build_dict)


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    eval_ppl(opt)
