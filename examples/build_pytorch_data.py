#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Generates a pytorch data file from the training data; for use in the
PytorchDataTeacher.

For more documentation, see parlai.scripts.build_pytorch_data.
"""
from parlai.scripts.build_pytorch_data import setup_args, build_data


if __name__ == '__main__':
    opt = setup_args().parse_args()
    build_data(opt)
