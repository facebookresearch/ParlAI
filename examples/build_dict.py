# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Generates a dictionary file from the training data."""

from parlai.scripts.build_dict import setup_args, build_dict


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    build_dict(opt)
