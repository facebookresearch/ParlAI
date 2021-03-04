#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch

from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser


def test_original_greedy_search(opt: Opt):

    agent = create_agent(opt, requireModelExists=True)
    # Using create_agent() instead of create_agent_from_model_file() because I couldn't get
    # --no-cuda to be recognized with the latter
    # get the tokenization
    agent.model.eval()
    obs = agent.observe({'text': opt['input'], 'episode_done': True})
    batch = agent.batchify([obs])
    tokens = batch.text_vec

    result = jit_greedy_search(model=agent.model, x=tokens)
    print(result)
    print(agent._v2t(result[0].tolist()))


def jit_greedy_search(model, x: torch.Tensor, max_len: int = 128):
    """
    A helper function for exporting simple greedy-search models via
    TorchScript.
    Models with extra inputs will need to override to include more
    variables.
    Utilize with:
    >>> TODO: write this
    """
    incr_state: Optional[Dict[int, Dict[str, Dict[str, torch.Tensor]]]] = None
    bsz = x.size(0)
    encoder_states = model.encoder(x)
    generations = model._get_initial_decoder_input(bsz, 1).to(x.device)
    # keep track of early stopping if all generations finish
    seen_end = torch.zeros(x.size(0), device=x.device, dtype=torch.bool)
    for timestep in range(max_len):
        latent, incr_state = model.decoder(generations, encoder_states, incr_state)
        logits = model.output(latent[:, -1:, :])
        _, preds = logits.max(dim=2)
        seen_end = seen_end + (preds == model.END_IDX).squeeze(1)
        generations = torch.cat([generations, preds], dim=1)
        if torch.all(seen_end):
            break
    return generations


def setup_args() -> ParlaiParser:
    # TODO: copied this from test_jit.py
    parser = ParlaiParser(add_parlai_args=True, add_model_args=True)
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='hello world',
        help='Test input string to pass into the encoder of the scripted model',
    )
    return parser


if __name__ == '__main__':
    parser = setup_args()
    opt_ = parser.parse_args()
    test_original_greedy_search(opt_)
