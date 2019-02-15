#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""
See parlai.scripts.display_model for documentation.
"""

from parlai.scripts.display_model import display_model, setup_args

if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    display_model(opt)
