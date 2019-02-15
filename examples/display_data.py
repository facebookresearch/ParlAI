#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which iterates through the tasks specified and prints them out.
Used for verification of data loading and iteration.

For more documentation, see parlai.scripts.display_data.
"""

from parlai.scripts.display_data import display_data, setup_args
import random


if __name__ == '__main__':
    random.seed(42)

    # Get command line arguments
    parser = setup_args()
    opt = parser.parse_args()
    display_data(opt)
