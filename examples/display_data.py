#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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
