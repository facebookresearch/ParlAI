#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Run the python or pytorch profiler and prints the results.

For documentation, see parlai.scripts.profile_train.
"""
from parlai.scripts.profile_train import setup_args, profile


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    profile(opt)
