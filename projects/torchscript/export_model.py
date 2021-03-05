#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch.jit
import torch.nn as nn

from parlai.core.agents import create_agent
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.io import PathManager


def test_jit(opt: Opt):

    agent = create_agent(opt, requireModelExists=True)
    # Using create_agent() instead of create_agent_from_model_file() because I couldn't get
    # --no-cuda to be recognized with the latter
    # get the tokenization
    agent.model.eval()
    inputs = agent.opt["input"].split("|")
    history_vecs = []
    delimiter_tok = agent.history.delimiter_tok
    if agent.opt.get('history_add_global_end_token', None) is not None:
        global_end_token = agent.dict[agent.dict.end_token]
    else:
        global_end_token = None
    bart = agent.opt['model'] == 'bart'
    text_truncate = agent.opt.get('text_truncate') or agent.opt['truncate']
    text_truncate = text_truncate if text_truncate >= 0 else None

    def _get_label_from_vec(label_vec_: torch.LongTensor) -> str:
        if bart:
            assert label_vec_[0, 0].item() == agent.END_IDX
            label_vec_ = label_vec_[:, 1:]
            # Hack: remove initial end token. I haven't found in the code where this is
            # done, but it seems to happen early on during generation
        return agent._v2t(label_vec_[0].tolist())

    # Script and trace the greedy search routine
    search_module = JitGreedySearch(agent.model)
    scripted_module = torch.jit.script(search_module)

    # Save the scripted module
    with PathManager.open(opt['scripted_model_file'], 'wb') as f:
        torch.jit.save(scripted_module, f)

    for input_ in inputs:

        # Vectorize this line of context
        print(" TEXT: " + input_)
        _update_vecs(
            history_vecs=history_vecs,
            size=agent.opt["history_size"],
            dict_=agent.dict,
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
        label_vec = scripted_module(batch_history_vec)
        label = _get_label_from_vec(label_vec)
        print("  SCRIPTED MODEL OUTPUT: " + label)
        _update_vecs(
            history_vecs=history_vecs,
            size=agent.opt["history_size"],
            dict_=agent.dict,
            text=label,
        )

        # Compare against the output from the unscripted model
        unscripted_label_vec = search_module(batch_history_vec)
        unscripted_label = _get_label_from_vec(unscripted_label_vec)
        print("UNSCRIPTED MODEL OUTPUT: " + unscripted_label)


def _update_vecs(history_vecs: List[int], size: int, dict_: DictionaryAgent, text: str):
    if size > 0:
        while len(history_vecs) >= size:
            history_vecs.pop(0)
    new_vec = list(dict_._word_lookup(token) for token in dict_.tokenize(str(text)))
    history_vecs.append(new_vec)


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
        latent, incr_state = model.decoder(initial_generations, encoder_states)
        logits = model.output(latent[:, -1:, :])
        _, preds = logits.max(dim=2)
        generations = torch.cat([initial_generations, preds], dim=1)

        # Do tracing
        self.encoder = torch.jit.trace(model.encoder, sample_tokens)
        self.decoder_first_pass = torch.jit.trace(
            model.decoder, (initial_generations, encoder_states), strict=False
        )
        # We do strict=False to avoid an error when passing a Dict out of
        # decoder.forward()
        self.decoder_later_pass = torch.jit.trace(
            model.decoder, (generations, encoder_states, incr_state), strict=False
        )
        self.partially_traced_model = torch.jit.trace_module(
            model,
            {
                'output': (latent[:, -1:, :]),
                'reorder_decoder_incremental_state': (
                    incr_state,
                    torch.LongTensor([0], device=sample_tokens.device),
                ),
            },
            strict=False,
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


if __name__ == '__main__':
    parser_ = setup_args()
    opt_ = parser_.parse_args()
    test_jit(opt_)
