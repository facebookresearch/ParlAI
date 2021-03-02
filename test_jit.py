#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch.jit
import torch.nn as nn

from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.utils.io import PathManager


def test_jit(opt):

    agent = create_agent(opt, requireModelExists=True)
    # Using create_agent() instead of create_agent_from_model_file() because I couldn't get
    # --no-cuda to be recognized with the latter
    # get the tokenization
    agent.model.eval()
    obs = agent.observe({'text': 'hello world', 'episode_done': True})
    batch = agent.batchify([obs])
    tokens = batch.text_vec

    # Script and trace the greedy search routine
    scripted_module = torch.jit.script(JitGreedySearch(agent.model))
    result = scripted_module(tokens)
    print(agent._v2t(result[0].tolist()))

    # Save the scripted module
    with PathManager.open(opt['scripted_model_file'], 'wb') as f:
        torch.jit.save(scripted_module, f)


def setup_args() -> ParlaiParser:
    parser = ParlaiParser(add_parlai_args=True, add_model_args=True)
    parser.add_argument(
        '-smf',
        '--scripted-model-file',
        type=str,
        default='_scripted.pt',
        help='Where the scripted model checkpoint will be saved',
    )
    return parser


class JitGreedySearch(nn.Module):
    """
    A helper class for exporting simple greedy-search models via TorchScript.

    Models with extra inputs will need to override to include more variables.

    Utilize with:

    >>> TODO: write this
    """

    def __init__(self, model):
        super().__init__()

        # Create sample inputs
        sample_tokens = torch.LongTensor([[1, 2, 3, 4, 5]])
        self.batch_size = 1
        self.num_dec_layers = model.decoder.n_layers
        self.emb_dim = model.decoder.embedding_size
        self.num_heads = model.decoder.n_heads
        self.dim_per_head = self.emb_dim // self.num_heads
        self.orig_incr_state_len = 0

        bsz = sample_tokens.size(0)
        encoder_states = model.encoder(sample_tokens)
        initial_generations = model._get_initial_decoder_input(bsz, 1).to(
            sample_tokens.device
        )
        latent, incr_state = model.decoder(initial_generations, encoder_states)
        logits = model.output(latent[:, -1:, :])
        _, preds = logits.max(dim=2)
        generations = torch.cat([initial_generations, preds], dim=1)

        self.encoder = torch.jit.trace(model.encoder, sample_tokens)
        self.decoder_first_pass = torch.jit.trace(
            model.decoder, (initial_generations, encoder_states), strict=False
        )
        # We do strict=False to avoid an error when passing a Dict out of
        # decoder.forward()
        self.partially_traced_model = torch.jit.trace_module(
            model, {'output': (latent[:, -1:, :])}
        )
        self.decoder_later_pass = torch.jit.trace(
            model.decoder, (generations, encoder_states, incr_state), strict=False
        )

        self.start_idx = model.START_IDX
        self.end_idx = model.END_IDX
        self.null_idx = model.NULL_IDX

    def forward(self, x: torch.Tensor, max_len: int = 128):
        bsz = x.size(0)
        encoder_states = self.encoder(x)
        generations = (
            (torch.ones(1, dtype=torch.long) * self.start_idx)
            .expand(bsz, 1)
            .to(x.device)
        )
        # Can't use TGM._get_initial_decoder_input() directly: when we do, we get a
        # "RuntimeError: Type 'Tuple[int, int]' cannot be traced. Only Tensors and
        # (possibly nested) Lists, Dicts, and Tuples of Tensors can be traced" error
        # keep track of early stopping if all generations finish
        seen_end = torch.zeros(x.size(0), device=x.device, dtype=torch.bool)
        incr_state: Dict[str, torch.Tensor] = {}
        for token_idx in range(max_len):
            if token_idx == 0:
                latent, incr_state = self.decoder_first_pass(
                    generations, encoder_states
                )
            else:
                latent, incr_state = self.decoder_later_pass(
                    generations, encoder_states, incr_state
                )
            logits = self.partially_traced_model.output(latent[:, -1:, :])
            _, preds = logits.max(dim=2)
            seen_end = seen_end + (preds == self.end_idx).squeeze(1)
            generations = torch.cat([generations, preds], dim=1)
            if torch.all(seen_end):
                break
        return generations


if __name__ == '__main__':
    parser = setup_args()
    opt_ = parser.parse_args()
    test_jit(opt_)
