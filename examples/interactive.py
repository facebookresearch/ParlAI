#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
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
