#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Generates a dictionary file from the training data.

For more documentation, see `parlai.scripts.build_dict`.
"""

from parlai.scripts.build_dict import setup_args, build_dict


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    build_dict(opt)
