#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch.jit
import torch.nn as nn

from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.utils.io import PathManager
from projects.torchscript.util import generate_given_module, setup_args


def export_model(opt: Opt):

    agent = create_agent(opt, requireModelExists=True)
    agent.model.eval()
    is_bart = agent.opt['model'] == 'bart'

    # Script and trace the greedy search routine
    original_module = JitGreedySearch(agent.model, bart=is_bart)
    scripted_module = torch.jit.script(original_module)

    # Save the scripted module
    with PathManager.open(opt['scripted_model_file'], 'wb') as f:
        torch.jit.save(scripted_module, f)

    inputs = opt['input'].split('|')

    print('\nGenerating given the scripted module:')
    generate_given_module(agent=agent, module=scripted_module, inputs=inputs)

    print('\nGenerating given the original unscripted module:')
    generate_given_module(agent=agent, module=original_module, inputs=inputs)


class JitGreedySearch(nn.Module):
    """
    A helper class for exporting simple greedy-search models via TorchScript.

    Models with extra inputs will need to override to include more variables.

    Utilize with:

    >>> TODO: write this
    """

    def __init__(self, model, bart: bool = False):
        super().__init__()

        self.start_idx = model.START_IDX
        self.end_idx = model.END_IDX
        self.null_idx = model.NULL_IDX
        if bart:
            self.initial_decoder_input = [self.end_idx, self.start_idx]
        else:
            self.initial_decoder_input = [self.start_idx]

        # Create sample inputs for tracing
        sample_tokens = torch.LongTensor([[1, 2, 3, 4, 5]])
        encoder_states = model.encoder(sample_tokens)
        initial_generations = self._get_initial_decoder_input(sample_tokens)
        latent, initial_incr_state = model.decoder(initial_generations, encoder_states)
        logits = model.output(latent[:, -1:, :])
        _, preds = logits.max(dim=2)
        incr_state = {k: torch.clone(v) for k, v in initial_incr_state.items()}
        # Copy the initial incremental state, used when tracing the
        # .reorder_decoder_incremental_state() method below, to avoid having it be
        # mutated by the following line
        incr_state = model.reorder_decoder_incremental_state(
            incr_state, torch.tensor([0], dtype=torch.long, device=sample_tokens.device)
        )
        generations = torch.cat([initial_generations, preds], dim=1)

        # Do tracing
        self.encoder = torch.jit.trace(model.encoder, sample_tokens)
        self.decoder_first_pass = torch.jit.trace(
            model.decoder, (initial_generations, encoder_states), strict=False
        )
        # We do strict=False to avoid an error when passing a Dict out of
        # decoder.forward()
        self.partially_traced_model = torch.jit.trace_module(
            model,
            {
                'output': (latent[:, -1:, :]),
                'reorder_decoder_incremental_state': (
                    initial_incr_state,
                    torch.LongTensor([0], device=sample_tokens.device),
                ),
            },
            strict=False,
        )
        self.decoder_later_pass = torch.jit.trace(
            model.decoder, (generations, encoder_states, incr_state), strict=False
        )

    def _get_initial_decoder_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Can't use TGM._get_initial_decoder_input() directly: when we do, we get a
        "RuntimeError: Type 'Tuple[int, int]' cannot be traced. Only Tensors and
        (possibly nested) Lists, Dicts, and Tuples of Tensors can be traced" error
        """
        bsz = x.size(0)
        return (
            torch.tensor(self.initial_decoder_input, dtype=torch.long)
            .expand(bsz, len(self.initial_decoder_input))
            .to(x.device)
        )

    def forward(self, x: torch.Tensor, max_len: int = 128):
        encoder_states = self.encoder(x)
        generations = self._get_initial_decoder_input(x)
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
            incr_state = self.partially_traced_model.reorder_decoder_incremental_state(
                incr_state, torch.tensor([0], dtype=torch.long, device=x.device)
            )
            seen_end = seen_end + (preds == self.end_idx).squeeze(1)
            generations = torch.cat([generations, preds], dim=1)
            if torch.all(seen_end):
                break
        return generations


if __name__ == '__main__':
    parser_ = setup_args()
    opt_ = parser_.parse_args()
    export_model(opt_)
