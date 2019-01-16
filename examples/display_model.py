#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
See parlai.scripts.display_model for documentation.
"""

from parlai.scripts.display_model import display_model, setup_args

if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    display_model(opt)
