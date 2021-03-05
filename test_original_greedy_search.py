#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
from parlai.core.agents import create_agent
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser


def test_original_greedy_search(opt: Opt):

    agent = create_agent(opt, requireModelExists=True)
    # Using create_agent() instead of create_agent_from_model_file() because I couldn't get
    # --no-cuda to be recognized with the latter
    # get the tokenization
    agent.model.eval()
    inputs = opt["input"].split("|")
    history_vecs = []
    delimiter_tok = agent.history.delimiter_tok
    if opt.get('history_add_global_end_token', None) is not None:
        global_end_token = agent.dict[agent.dict.end_token]
    else:
        global_end_token = None
    bart = opt['model'] == 'bart'
    text_truncate = opt.get('text_truncate') or opt['truncate']
    text_truncate = text_truncate if text_truncate >= 0 else None

    for input_ in inputs:

        # Vectorize this line of context
        print(" TEXT: " + input_)
        _update_vecs(
            history_vecs=history_vecs,
            size=opt["history_size"],
            dict=agent.dict,
            text=input_,
        )

        # Get full history vec
        full_history_vec = []
        for vec in history_vecs[:-1]:
            full_history_vec += [vec]
            full_history_vec += [delimiter_tok]
        full_history_vec += [history_vecs[-1]]
        if global_end_token is not None:
            full_history_vec += [[global_end_token]]
        full_history_vec = sum(full_history_vec, [])

        # Format history vec given various logic
        if text_truncate is not None:
            if bart:
                truncate_length = text_truncate - 2  # Start and end tokens
            else:
                truncate_length = text_truncate
            if len(full_history_vec) > truncate_length:
                full_history_vec = full_history_vec[-truncate_length:]
        full_history_vec = torch.LongTensor(full_history_vec)
        if bart:
            full_history_vec = torch.cat(
                [
                    full_history_vec.new_tensor([agent.START_IDX]),
                    full_history_vec,
                    full_history_vec.new_tensor([agent.END_IDX]),
                ],
                axis=0,
            )

        # Use greedy search to get model response
        batch_history_vec = torch.unsqueeze(full_history_vec, dim=0)  # Add batch dim
        label_vec = jit_greedy_search(agent=agent, x=batch_history_vec)
        if bart:
            assert label_vec[0, 0].item() == agent.END_IDX
            label_vec = label_vec[:, 1:]
            # Hack: remove initial end token. I haven't found in the code where this is
            # done, but it seems to happen early on during generation
        label = agent._v2t(label_vec[0].tolist())
        print("LABEL: " + label)
        _update_vecs(
            history_vecs=history_vecs,
            size=opt["history_size"],
            dict=agent.dict,
            text=label,
        )


def _update_vecs(history_vecs: List[int], size: int, dict: DictionaryAgent, text: str):
    if size > 0:
        while len(history_vecs) >= size:
            history_vecs.pop(0)
    new_vec = list(dict._word_lookup(token) for token in dict.tokenize(str(text)))
    history_vecs.append(new_vec)


def jit_greedy_search(agent, x: torch.Tensor, max_len: int = 128):
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
    encoder_states = agent.model.encoder(x)
    generations = agent._get_initial_decoder_input(bsz, 1, x.device)
    # keep track of early stopping if all generations finish
    seen_end = torch.zeros(x.size(0), device=x.device, dtype=torch.bool)
    for timestep in range(max_len):
        latent, incr_state = agent.model.decoder(
            generations, encoder_states, incr_state
        )
        logits = agent.model.output(latent[:, -1:, :])
        _, preds = logits.max(dim=2)
        incr_state = agent.model.reorder_decoder_incremental_state(
            incr_state, inds=torch.LongTensor([0], device=x.device)
        )
        seen_end = seen_end + (preds == agent.model.END_IDX).squeeze(1)
        generations = torch.cat([generations, preds], dim=1)
        if torch.all(seen_end):
            break
    return generations


def setup_args() -> ParlaiParser:
    # TODO: copied this from test_jit.py
    parser = ParlaiParser(add_parlai_args=True, add_model_args=True)
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="hello world",
        help="Test input string to pass into the encoder of the scripted model. Separate lines with a pipe",
    )
    return parser


if __name__ == "__main__":
    parser = setup_args()
    opt_ = parser.parse_args()
    test_original_greedy_search(opt_)
