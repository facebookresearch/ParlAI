#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Basic example which iterates through the tasks specified and
evaluates the given model on them.

For more documentation, see parlai.scripts.eval_model.
"""
from parlai.scripts.eval_model import setup_args, eval_model


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args(print_args=False)
    eval_model(opt, print_parser=parser)
