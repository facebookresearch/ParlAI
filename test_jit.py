#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.torch_generator_agent import TorchGeneratorModel


def test_jit(opt):

    agent = create_agent(opt, requireModelExists=True)
    # Using create_agent() instead of create_agent_from_model_file() because I couldn't get
    # --no-cuda to be recognized with the latter
    # get the tokenization
    obs = agent.observe({'text': 'hello world', 'episode_done': True})
    batch = agent.batchify([obs])
    tokens = batch.text_vec

    # Script and trace the greedy search routine
    scripted_module = torch.jit.script(JitGreedySearch(agent.model))
    result = scripted_module(tokens)
    print(agent._v2t(result[0].tolist()))

    # Save the scripted module
    scripted_module.save('_scripted.pt')


class JitGreedySearch(nn.Module):
    """
    A helper class for exporting simple greedy-search models via TorchScript.

    Models with extra inputs will need to override to include more variables.

    Utilize with:

    >>> TODO: write this
    """

    def __init__(self, model):
        super().__init__()
        sample_tokens = torch.LongTensor([[1, 2, 3, 4, 5]])
        bsz = sample_tokens.size(0)
        encoder_states = model.encoder(sample_tokens)
        generations = model._get_initial_decoder_input(bsz, 1).to(sample_tokens.device)
        # latent, incr_state = model.decoder(generations, encoder_states, incr_state)
        latent = model.decoder(generations, encoder_states)
        self.encoder = torch.jit.trace(model.encoder, sample_tokens)
        self.decoder = torch.jit.trace(model.decoder, (generations, encoder_states))
        self.partially_traced_model = torch.jit.trace_module(
            model, {'output': (latent[:, -1:, :])}
        )
        self.start_idx = model.START_IDX
        self.end_idx = model.END_IDX
        self.null_idx = model.NULL_IDX

    def forward(self, x: torch.Tensor, max_len: int = 128):
        # incr_state: Optional[Dict[int, Dict[str, Dict[str, torch.Tensor]]]] = None
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
        for _ in range(max_len):
            # latent, incr_state = self.decoder(generations, encoder_states, incr_state)
            latent = self.decoder(generations, encoder_states)
            logits = self.partially_traced_model.output(latent[:, -1:, :])
            _, preds = logits.max(dim=2)
            seen_end = seen_end + (preds == self.end_idx).squeeze(1)
            generations = torch.cat([generations, preds], dim=1)
            if torch.all(seen_end):
                break
        padded_generations = F.pad(
            generations,
            pad=(0, max_len - generations.size(1)),
            value=float(self.null_idx),
        )
        # Just pad the generation to max_len so that the generation will be the same
        # size before and after tracing, which is needed when the tracer checks the
        # similarity of the outputs after tracing. The `value` arg needs to be a float
        # for some reason
        return padded_generations


if __name__ == '__main__':
    parser = ParlaiParser(add_parlai_args=True, add_model_args=True)
    opt_ = parser.parse_args()
    test_jit(opt_)
