#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch.jit
from parlai.agents.bart.bart import BartAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.torch_agent import TorchAgent
from parlai.utils.bpe import Gpt2BpeHelper, SubwordBPEHelper
from parlai.torchscript.tokenizer import ScriptableDictionaryAgent
from torch import nn as nn


class TorchScriptGreedySearch(nn.Module):
    """
    A helper class for exporting simple greedy-search models via TorchScript.

    Models with extra inputs will need to override to include more variables.
    """

    # We currently only support these specific dictionary settings
    CAIRAOKE_DICT_PARAMS = {
        "dict_class": "parlai.core.dict:DictionaryAgent",
        "dict_initpath": None,
        "dict_language": "english",
        "dict_max_ngram_size": -1,
        "dict_minfreq": 0,
        "dict_maxtokens": -1,
        "dict_tokenizer": ["gpt2", "slow_bytelevel_bpe"],
        "dict_textfields": "text,labels",
        "dict_loaded": True,
        "bpe_debug": False,
    }

    def __init__(self, agent: TorchAgent):
        super().__init__()

        self.is_bart = isinstance(agent, BartAgent)
        self.device = agent.model.encoder.embeddings.weight.device
        # Dictionary/tokenization setup
        for key, val in self.CAIRAOKE_DICT_PARAMS.items():
            if type(val) == list and len(val) > 0:
                assert (
                    agent.opt.get(key, val[0]) in val
                ), f'The only currently supported values of "{key}" are {", ".join(val)}!'
            else:
                assert (
                    agent.opt.get(key, val) == val
                ), f'The only currently supported value of "{key}" is {val}!'
        orig_dict: DictionaryAgent = agent.dict
        orig_bpe: Gpt2BpeHelper = orig_dict.bpe
        assert all(len(key) == 2 for key in orig_bpe.bpe_ranks.keys())
        if any(i for key in orig_bpe.bpe_ranks.keys() for i in key if "\n" in i):
            raise AssertionError(
                "We need to temporarily merge the bpe_ranks dict's keys with a newline "
                "character in order to use it as a TorchScript arg, but at least one of the "
                "dict's keys contains a newline character already!"
            )
        fused_key_bpe_ranks = {
            "\n".join(key): float(val) for key, val in orig_bpe.bpe_ranks.items()
        }
        # Cast the values as floats to be able to compare to float('inf') when doing BPE
        # splitting
        self.dict = ScriptableDictionaryAgent(
            null_token=orig_dict.null_token,
            end_token=orig_dict.end_token,
            unk_token=orig_dict.unk_token,
            start_token=orig_dict.start_token,
            freq=orig_dict.freq,
            tok2ind=orig_dict.tok2ind,
            ind2tok=orig_dict.ind2tok,
            bpe_add_prefix_space=agent.opt["bpe_add_prefix_space"],
            bpe_encoder=orig_bpe.encoder,
            bpe_byte_encoder=orig_bpe.byte_encoder,
            fused_key_bpe_ranks=fused_key_bpe_ranks,
            special_tokens=agent._get_special_tokens(),
            subword_bpe_version=(0, 0),
            fused_bpe_codes={},
            subword_bpe_separator='',
        )

        # History tracking and start/end tokens
        self.delimiter_tok = agent.history.delimiter_tok
        self.history_size = agent.opt["history_size"]
        if agent.opt.get("history_add_global_end_token", None) is not None:
            self.global_end_token = agent.dict[agent.dict.end_token]
        else:
            self.global_end_token = None
        self.text_truncate = agent.opt.get("text_truncate") or agent.opt["truncate"]
        self.text_truncate = self.text_truncate if self.text_truncate >= 0 else None

        self.start_idx = agent.model.START_IDX
        self.end_idx = agent.model.END_IDX
        self.null_idx = agent.model.NULL_IDX
        if self.is_bart:
            self.initial_decoder_input = [self.end_idx, self.start_idx]
        else:
            self.initial_decoder_input = [self.start_idx]

        agent.model.eval()

        # Create versions of the model and decoder that will flatten the incremental
        # state dict, as required by TorchScript
        wrapped_decoder = DecoderIncrStateFlattener(agent.model.decoder)
        wrapped_model = ModelIncrStateFlattener(agent.model)

        # Create sample inputs for tracing
        sample_tokens = torch.tensor(
            [[1, 2, 3, 4, 5]], dtype=torch.long, device=self.device
        )
        sample_tokens = sample_tokens.to(self.device)
        encoder_states = agent.model.encoder(sample_tokens)
        initial_generations = self._get_initial_decoder_input(sample_tokens)
        latent, initial_incr_state = wrapped_decoder(
            initial_generations, encoder_states
        )
        logits = agent.model.output(latent[:, -1:, :])
        _, preds = logits.max(dim=2)
        incr_state = {k: torch.clone(v) for k, v in initial_incr_state.items()}
        # Copy the initial incremental state, used when tracing the
        # .reorder_decoder_incremental_state() method below, to avoid having it be
        # mutated by the following line
        incr_state = wrapped_model.reorder_decoder_incremental_state(
            incr_state, torch.tensor([0], dtype=torch.long, device=sample_tokens.device)
        )
        generations = torch.cat([initial_generations, preds], dim=1)

        # Do tracing
        self.encoder = torch.jit.trace(agent.model.encoder, sample_tokens)
        self.decoder_first_pass = torch.jit.trace(
            wrapped_decoder, (initial_generations, encoder_states), strict=False
        )
        # We do strict=False to avoid an error when passing a Dict out of
        # decoder.forward()
        self.partially_traced_model = torch.jit.trace_module(
            wrapped_model,
            {
                "output": (latent[:, -1:, :]),
                "reorder_decoder_incremental_state": (
                    initial_incr_state,
                    torch.tensor([0], dtype=torch.long, device=sample_tokens.device),
                ),
            },
            strict=False,
        )
        self.decoder_later_pass = torch.jit.trace(
            wrapped_decoder, (generations, encoder_states, incr_state), strict=False
        )

    def get_device(self):
        return self.encoder.embeddings.weight.device

    def _get_initial_decoder_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Workaround because we can't use TGM._get_initial_decoder_input() directly.

        When we try to call that function, we get a "RuntimeError: Type 'Tuple[int,
        int]' cannot be traced. Only Tensors and (possibly nested) Lists, Dicts, and
        Tuples of Tensors can be traced" error.
        """
        bsz = x.size(0)
        return (
            torch.tensor(
                self.initial_decoder_input, dtype=torch.long, device=self.device
            )
            .expand(bsz, len(self.initial_decoder_input))
            .to(x.device)
        )

    def parse(self, text: str) -> List[int]:
        return self.dict.txt2vec(text, dict_tokenizer='gpt2')

    def _v2t(self, vec: List[int]) -> str:
        """
        Convert token indices to string of tokens.
        """
        new_vec: List[int] = []
        for i in vec:
            if i == self.end_idx:
                break
            elif i != self.start_idx:
                new_vec.append(i)
        return self.dict.vec2txt(new_vec, dict_tokenizer='gpt2')

    def forward(self, context: str, max_len: int = 128) -> str:

        # Vectorize all lines of context
        history_vecs: List[List[int]] = []
        context_lines = context.split("\n")
        context_lines = self.preprocess_context(context_lines)
        if self.history_size > 0:
            context_lines = context_lines[-self.history_size :]
        for line in context_lines:
            history_vecs.append(self.parse(line))

        # Get full history vec
        text_vecs: List[List[int]] = []
        for vec in history_vecs[:-1]:
            text_vecs += [vec]
            text_vecs += [self.delimiter_tok]
        text_vecs += [history_vecs[-1]]
        if self.global_end_token is not None:
            text_vecs += [[self.global_end_token]]

        # Flatten text_vecs
        flattened_text_vec: List[int] = []
        for vec in text_vecs:
            for token in vec:
                flattened_text_vec.append(token)

        # Format history vec given various logic
        if self.text_truncate is not None:
            if self.is_bart:
                truncate_length = self.text_truncate - 2  # Start and end tokens
            else:
                truncate_length = self.text_truncate
            if len(flattened_text_vec) > truncate_length:
                flattened_text_vec = flattened_text_vec[-truncate_length:]
        flattened_text_vec = torch.tensor(flattened_text_vec, dtype=torch.long)
        if self.is_bart:
            flattened_text_vec = torch.cat(
                [
                    torch.tensor([self.start_idx], dtype=torch.long).to(
                        self.get_device()
                    ),
                    flattened_text_vec.to(self.get_device()),
                    torch.tensor([self.end_idx], dtype=torch.long).to(
                        self.get_device()
                    ),
                ],
                dim=0,
            )

        # Pass through the encoder and decoder to generate tokens

        flattened_text_vec = flattened_text_vec.to(self.get_device())
        batch_text_vec = torch.unsqueeze(flattened_text_vec, dim=0)  # Add batch dim
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
        generation_tokens: List[int] = generations[0].tolist()
        label = self._v2t(generation_tokens)

        return self.postprocess_output_generations(label=label)

    def postprocess_output_generations(self, label: str) -> str:
        """
        Post-process the model output.

        Returns the model output by default, override to add custom logic
        """
        return label

    def preprocess_context(self, context_lines: List[str]) -> List[str]:
        return context_lines


class BaseIncrStateFlattener(nn.Module):
    """
    Flatten/unflatten the incremental state for use with TorchScripting.

    Typically, the incremental state will be stored as a Dict[int, Dict[str, Dict[str,
    torch.Tensor]]], where the 3 dictionary levels map decoder layer, attention type,
    and previous key/value/mask, respectively. However, TorchScript expects dicts to be
    of type Dict[str, torch.Tensor], and thus all input incremental states when
    TorchScripting will have to be of that type. We thus unflatten the input incremental
    state, already of type Dict[str, torch.Tensor], to pass it into whatever method
    needs it, and we flatten it again after the updated incremental state is passed back
    out.

    This is a base class that provides methods for flattening/unflattening: subclasses
    will call these methods as the incremental state is passed into and out of their own
    methods.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def _unflatten_incr_state(
        self, flat_incr_state: Dict[str, torch.Tensor]
    ) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Unflatten the input incremental state.

        For instance, flat_incr_state['layer_0__self_attn__prev_key'] will be stored in
        structured_incr_state[0]['self_attn']['prev_key'].
        """
        structured_incr_state = defaultdict(lambda: defaultdict(dict))
        for key, state in flat_incr_state.items():
            layer_idx_str, attn_type, state_type = key.split("__")
            structured_incr_state[int(layer_idx_str)][attn_type][state_type] = state
        return dict({k: dict(v) for k, v in structured_incr_state.items()})
        # Turn the nested defaultdicts back into regular dicts

    def _flatten_incr_state(
        self, structured_incr_state: Dict[int, Dict[str, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Flatten the input incremental state.

        For instance, structured_incr_state[0]['self_attn']['prev_key'] will be stored
        in flat_incr_state['layer_0__self_attn__prev_key'].
        """
        flat_incr_state = {}
        for layer_idx, dict1 in structured_incr_state.items():
            for attn_type, dict2 in dict1.items():
                for state_type, state in dict2.items():
                    key = f"{layer_idx:d}__{attn_type}__{state_type}"
                    flat_incr_state[key] = state
        return flat_incr_state


class DecoderIncrStateFlattener(BaseIncrStateFlattener):
    """
    Wrapper for a TransformerDecoder that will unflatten/flatten the incremental state.

    Unflattening/flattening will occur before passing the incremental state into and out
    of .forward().
    """

    def forward(
        self,
        input_: torch.LongTensor,
        encoder_state: Tuple[torch.Tensor, torch.Tensor],
        flat_incr_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if flat_incr_state is not None:
            structured_incr_state = self._unflatten_incr_state(flat_incr_state)
        else:
            structured_incr_state = None
        tensor, new_structured_incr_state = self.module.forward(
            input=input_, encoder_state=encoder_state, incr_state=structured_incr_state
        )
        new_flat_incr_state = self._flatten_incr_state(new_structured_incr_state)
        return tensor, new_flat_incr_state


class ModelIncrStateFlattener(BaseIncrStateFlattener):
    """
    Wrapper for a TransformerGeneratorModel to unflatten/flatten the incremental state.

    Unflattening/flattening will occur before passing the incremental state into and out
    of .reorder_decoder_incremental_state(). We also support .output(), which is also
    traced.
    """

    def reorder_decoder_incremental_state(
        self, flat_incr_state: Dict[str, torch.Tensor], inds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        structured_incr_state = self._unflatten_incr_state(flat_incr_state)
        new_structured_incr_state = self.module.reorder_decoder_incremental_state(
            incremental_state=structured_incr_state, inds=inds
        )
        return self._flatten_incr_state(new_structured_incr_state)

    def output(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.module.output(tensor)


class TorchScriptTransformerClassifier(nn.Module):
    """
    A helper class for exporting transformer classifier via TorchScript.
    """

    def __init__(self, agent: TorchAgent):
        super().__init__()
        self.device = agent.model.transformer.embeddings.weight.device
        self.start_idx = agent.START_IDX
        self.end_idx = agent.END_IDX

        orig_dict: DictionaryAgent = agent.dict
        orig_bpe: SubwordBPEHelper = orig_dict.bpe

        assert all(len(key) == 2 for key in orig_bpe.bpe.bpe_codes.keys())
        if any(i for key in orig_bpe.bpe.bpe_codes.keys() for i in key if "\n" in i):
            raise AssertionError(
                "We need to temporarily merge the bpe_codes dict's keys with a newline "
                "character in order to use it as a TorchScript arg, but at least one of the "
                "dict's keys contains a newline character already!"
            )
        fused_key_bpe_codes = {
            "\n".join(key): float(val) for key, val in orig_bpe.bpe.bpe_codes.items()
        }

        # Initialize a subword Bpe tokenizer
        self.dict = ScriptableDictionaryAgent(
            null_token=orig_dict.null_token,
            end_token=orig_dict.end_token,
            unk_token=orig_dict.unk_token,
            start_token=orig_dict.start_token,
            freq=orig_dict.freq,
            tok2ind=orig_dict.tok2ind,
            ind2tok=orig_dict.ind2tok,
            bpe_add_prefix_space=agent.opt["bpe_add_prefix_space"],
            bpe_encoder={},
            bpe_byte_encoder={},
            fused_key_bpe_ranks={},
            special_tokens=agent._get_special_tokens(),
            subword_bpe_version=orig_bpe.bpe.version,
            fused_bpe_codes=fused_key_bpe_codes,
            subword_bpe_separator=orig_bpe.bpe.separator,
        )

        self.delimiter_tok = agent.history.delimiter_tok
        self.history_size = agent.opt["history_size"]
        if agent.opt.get("history_add_global_end_token", None) is not None:
            self.global_end_token = agent.dict[agent.dict.end_token]
        else:
            self.global_end_token = None
        self.text_truncate = agent.opt.get("text_truncate") or agent.opt["truncate"]
        self.text_truncate = self.text_truncate if self.text_truncate >= 0 else None
        self.class_list = agent.class_list

        agent.model.eval()
        # Create sample inputs for tracing
        sample_tokens = torch.tensor(
            [[1, 2, 3, 4, 5]], dtype=torch.long, device=self.device
        )
        scores = agent.model(sample_tokens)
        _, prediction = torch.max(scores, 1)

        # Do tracing
        self.model = torch.jit.trace(agent.model, sample_tokens)

    def get_device(self):
        return self.model.transformer.embeddings.weight.device

    def parse(self, text: str) -> List[int]:
        return self.dict.txt2vec(text, dict_tokenizer='bpe')

    def forward(self, context: str, max_len: int = 128) -> str:

        history_vecs: List[List[int]] = []
        context_lines = context.split("\n")
        for line in context_lines:
            history_vecs.append(self.parse(line))
        if self.history_size > 0:
            context_lines = context_lines[-self.history_size :]
        # Get full history vec
        text_vecs: List[List[int]] = []
        for vec in history_vecs[:-1]:
            text_vecs += [vec]
            text_vecs += [self.delimiter_tok]
        text_vecs += [history_vecs[-1]]
        if self.global_end_token is not None:
            text_vecs += [[self.global_end_token]]
        # Flatten text_vecs
        flattened_text_vec: List[int] = []
        for vec in text_vecs:
            for token in vec:
                flattened_text_vec.append(token)

        if self.text_truncate is not None:
            truncate_length = self.text_truncate
            if len(flattened_text_vec) > truncate_length:
                flattened_text_vec = flattened_text_vec[-truncate_length:]
        flattened_text_vec = torch.tensor(flattened_text_vec, dtype=torch.long)

        # Added start and end token idx
        flattened_text_vec = torch.cat(
            [
                torch.tensor([self.start_idx], dtype=torch.long).to(self.get_device()),
                flattened_text_vec.to(self.get_device()),
                torch.tensor([self.end_idx], dtype=torch.long).to(self.get_device()),
            ],
            dim=0,
        )
        flattened_text_vec = flattened_text_vec.to(self.get_device())
        batch_text_vec = torch.unsqueeze(flattened_text_vec, dim=0)
        scores = self.model(batch_text_vec)
        _, prediction_id = torch.max(scores, 1)
        preds = self.class_list[prediction_id.squeeze()]

        return preds
