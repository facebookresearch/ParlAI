#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Basic example which iterates through the tasks specified and load/extract
the image features.

For more documentation, use parlai.scripts.extract_image_feature.
"""
from parlai.scripts.extract_image_feature import setup_args, extract_feats

if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    extract_feats(opt)
