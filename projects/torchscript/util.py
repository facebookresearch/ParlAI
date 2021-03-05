#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn

from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import TorchAgent


def setup_args() -> ParlaiParser:
    parser = ParlaiParser(add_parlai_args=True, add_model_args=True)
    parser.add_argument(
        '-smf',
        '--scripted-model-file',
        type=str,
        default='_scripted.pt',
        help='Where the scripted model checkpoint will be saved',
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="hello world",
        help="Test input string to pass into the encoder of the scripted model. Separate lines with a pipe",
    )
    return parser


def generate_given_module(agent: TorchAgent, module: nn.Module, inputs: List[str]):

    is_bart = agent.opt['model'] == 'bart'
    history_vecs = []
    delimiter_tok = agent.history.delimiter_tok
    history_size = agent.opt['history_size']
    if agent.opt.get('history_add_global_end_token', None) is not None:
        global_end_token = agent.dict[agent.dict.end_token]
    else:
        global_end_token = None
    text_truncate = agent.opt.get('text_truncate') or agent.opt['truncate']
    text_truncate = text_truncate if text_truncate >= 0 else None

    def _get_label_from_vec(label_vec_: torch.LongTensor) -> str:
        if is_bart:
            assert label_vec_[0, 0].item() == agent.END_IDX
            label_vec_ = label_vec_[:, 1:]
            # Hack: remove initial end token. I haven't found in the code where this is
            # done, but it seems to happen early on during generation
        return agent._v2t(label_vec_[0].tolist())

    def _update_vecs(history_vecs_: List[int], text: str):
        if history_size > 0:
            while len(history_vecs_) >= history_size:
                history_vecs_.pop(0)
        new_vec = list(
            agent.dict._word_lookup(token) for token in agent.dict.tokenize(str(text))
        )
        history_vecs_.append(new_vec)

    for input_ in inputs:

        # Vectorize this line of context
        print(" TEXT: " + input_)
        _update_vecs(history_vecs_=history_vecs, text=input_)

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
            if is_bart:
                truncate_length = text_truncate - 2  # Start and end tokens
            else:
                truncate_length = text_truncate
            if len(full_history_vec) > truncate_length:
                full_history_vec = full_history_vec[-truncate_length:]
        full_history_vec = torch.LongTensor(full_history_vec)
        if is_bart:
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
        label_vec = module(batch_history_vec)
        label = _get_label_from_vec(label_vec)
        print("LABEL: " + label)
        _update_vecs(history_vecs_=history_vecs, text=label)
