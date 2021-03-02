#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.jit

from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.io import PathManager


def test_jit_exported_model(opt: Opt):

    with PathManager.open(opt["scripted_model_file"], "rb") as f:
        scripted_module = torch.jit.load(f)

    # Create the agent only to get the tokens, not to do the forward pass
    agent = create_agent(opt, requireModelExists=True)
    obs = agent.observe({"text": "hello world", "episode_done": True})
    batch = agent.batchify([obs])
    tokens = batch.text_vec

    result = scripted_module(tokens)
    print(agent._v2t(result[0].tolist()))


def setup_args() -> ParlaiParser:
    # TODO: deduplicate this function from test_jit.py
    parser = ParlaiParser(add_parlai_args=True, add_model_args=True)
    parser.add_argument(
        "-smf",
        "--scripted-model-file",
        type=str,
        default="_scripted.pt",
        help="Where the scripted model checkpoint will be saved",
    )
    return parser


if __name__ == "__main__":
    parser = setup_args()
    opt_ = parser.parse_args()
    test_jit_exported_model(opt_)
