# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Generates a pytorch data file from the training data; for use in the
PytorchDataTeacher.

Note that with our given implementation of batch act, episodes are compressed
such that each episode is one example for a model.

One can set the `--context-len` flag to specify how many past utterances
are used in a flattened episode

"""
from parlai.scripts.build_pytorch_data import setup_args, build_data


if __name__ == '__main__':
    opt = setup_args().parse_args()
    build_data(opt)
