#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run the python or pytorch profiler and prints the results.

For documentation, see parlai.scripts.profile_train.
"""
from parlai.scripts.profile_train import setup_args, profile


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    profile(opt)
