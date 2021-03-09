#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch.jit
import torch.nn as nn

from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.torch_agent import History, TorchAgent
from parlai.torchscript.util import setup_args
from parlai.utils.io import PathManager


def export_model(opt: Opt):

    agent = create_agent(opt, requireModelExists=True)

    # Script and trace the greedy search routine
    original_module = JitGreedySearch(agent)
    scripted_module = torch.jit.script(original_module)

    # Save the scripted module
    with PathManager.open(opt['scripted_model_file'], 'wb') as f:
        torch.jit.save(scripted_module, f)

    inputs = opt['input'].split('|')

    print('\nGenerating given the scripted module:')
    for input_ in inputs:
        label = scripted_module(input_)
        print("LABEL: " + label)

    print('\nGenerating given the original unscripted module:')
    for input_ in inputs:
        label = original_module(input_)
        print("LABEL: " + label)


class JitGreedySearch(nn.Module):
    """
    A helper class for exporting simple greedy-search models via TorchScript.

    Models with extra inputs will need to override to include more variables.

    Utilize with:

    >>> TODO: write this
    """

    def __init__(self, agent: TorchAgent):
        super().__init__()

        self.is_bart = agent.opt['model'] == 'bart'

        # Tokenization and history tracking
        self.history = History(
            agent.opt,
            maxlen=agent.text_truncate,
            size=agent.histsz,
            p1_token=agent.P1_TOKEN,
            p2_token=agent.P2_TOKEN,
            dict_agent=agent.dict,
        )
        self.text_truncate = agent.opt.get('text_truncate') or agent.opt['truncate']
        self.text_truncate = self.text_truncate if self.text_truncate >= 0 else None
        self.v2t = agent._v2t

        self.start_idx = agent.model.START_IDX
        self.end_idx = agent.model.END_IDX
        self.null_idx = agent.model.NULL_IDX
        if self.is_bart:
            self.initial_decoder_input = [self.end_idx, self.start_idx]
        else:
            self.initial_decoder_input = [self.start_idx]

        agent.model.eval()

        # Create sample inputs for tracing
        sample_tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        encoder_states = agent.model.encoder(sample_tokens)
        initial_generations = self._get_initial_decoder_input(sample_tokens)
        latent, initial_incr_state = agent.model.decoder(
            initial_generations, encoder_states
        )
        logits = agent.model.output(latent[:, -1:, :])
        _, preds = logits.max(dim=2)
        incr_state = {k: torch.clone(v) for k, v in initial_incr_state.items()}
        # Copy the initial incremental state, used when tracing the
        # .reorder_decoder_incremental_state() method below, to avoid having it be
        # mutated by the following line
        incr_state = agent.model.reorder_decoder_incremental_state(
            incr_state, torch.tensor([0], dtype=torch.long, device=sample_tokens.device)
        )
        generations = torch.cat([initial_generations, preds], dim=1)

        # Do tracing
        self.encoder = torch.jit.trace(agent.model.encoder, sample_tokens)
        self.decoder_first_pass = torch.jit.trace(
            agent.model.decoder, (initial_generations, encoder_states), strict=False
        )
        # We do strict=False to avoid an error when passing a Dict out of
        # decoder.forward()
        self.partially_traced_model = torch.jit.trace_module(
            agent.model,
            {
                'output': (latent[:, -1:, :]),
                'reorder_decoder_incremental_state': (
                    initial_incr_state,
                    torch.tensor([0], dtype=torch.long, device=sample_tokens.device),
                ),
            },
            strict=False,
        )
        self.decoder_later_pass = torch.jit.trace(
            agent.model.decoder, (generations, encoder_states, incr_state), strict=False
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

    def forward(self, input_: str, max_len: int = 128) -> str:
        # TODO: docstring

        # Vectorize this line of context
        print(" TEXT: " + input_)
        self.history.update_history(obs={'text': input_})
        # Forgo wrapping input as a Message to avoid having to TorchScript that type

        # Get full history vec
        text_vec = self.history.get_history_vec()

        # Format history vec given various logic
        if self.text_truncate is not None:
            if self.is_bart:
                truncate_length = self.text_truncate - 2  # Start and end tokens
            else:
                truncate_length = self.text_truncate
            if len(text_vec) > truncate_length:
                text_vec = text_vec[-truncate_length:]
        text_vec = torch.tensor(text_vec, dtype=torch.long)
        if self.is_bart:
            text_vec = torch.cat(
                [
                    text_vec.new_tensor([self.start_idx]),
                    text_vec,
                    text_vec.new_tensor([self.end_idx]),
                ],
                axis=0,
            )

        # Pass through the encoder and decoder to generate tokens
        batch_text_vec = torch.unsqueeze(text_vec, dim=0)  # Add batch dim
        encoder_states = self.encoder(batch_text_vec)
        generations = self._get_initial_decoder_input(batch_text_vec)
        # keep track of early stopping if all generations finish
        seen_end = torch.zeros(
            batch_text_vec.size(0), device=batch_text_vec.device, dtype=torch.bool
        )
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
                incr_state,
                torch.tensor([0], dtype=torch.long, device=batch_text_vec.device),
            )
            seen_end = seen_end + (preds == self.end_idx).squeeze(1)
            generations = torch.cat([generations, preds], dim=1)
            if torch.all(seen_end):
                break

        # Get the label from the generated tokens and update the history
        if self.is_bart:
            assert generations[0, 0].item() == self.end_idx
            generations = generations[:, 1:]
            # Hack: remove initial end token. I haven't found in the code where this is
            # done, but it seems to happen early on during generation
        label = self.v2t(generations[0].tolist())
        self.history.add_reply(text=label)

        return label


if __name__ == '__main__':
    parser_ = setup_args()
    opt_ = parser_.parse_args()
    export_model(opt_)
