#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""Basic example which iterates through the tasks specified and load/extract
the image features.

For more documentation, use parlai.scripts.extract_image_feature.
"""
from parlai.scripts.extract_image_feature import setup_args, extract_feats

if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    extract_feats(opt)
