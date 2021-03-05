#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.jit

from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.utils.io import PathManager
from projects.torchscript.util import generate_given_module, setup_args


def test_exported_model(opt: Opt):

    agent = create_agent(opt, requireModelExists=True)
    agent.model.eval()

    with PathManager.open(opt["scripted_model_file"], "rb") as f:
        scripted_module = torch.jit.load(f)

    inputs = opt['input'].split('|')

    print('\nGenerating given the scripted module:')
    generate_given_module(agent=agent, module=scripted_module, inputs=inputs)


if __name__ == '__main__':
    parser_ = setup_args()
    opt_ = parser_.parse_args()
    test_exported_model(opt_)
