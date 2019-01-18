#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Basic example which allows local human keyboard input to talk to a trained model.

For documentation, see parlai.scripts.interactive.
"""
from parlai.scripts.interactive import setup_args, interactive
import random


if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()
    opt = parser.parse_args()
    interactive(opt, print_parser=parser)
