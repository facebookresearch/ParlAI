#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.jit

from parlai.core.opt import Opt
from parlai.torchscript.util import setup_args
from parlai.utils.io import PathManager


def test_exported_model(opt: Opt):

    with PathManager.open(opt["scripted_model_file"], "rb") as f:
        scripted_module = torch.jit.load(f)

    inputs = opt['input'].split('|')

    print('\nGenerating given the scripted module:')
    for input_ in inputs:
        label = scripted_module(input_)
        print("LABEL: " + label)


if __name__ == '__main__':
    parser_ = setup_args()
    opt_ = parser_.parse_args()
    test_exported_model(opt_)
