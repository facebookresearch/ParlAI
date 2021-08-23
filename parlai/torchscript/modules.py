#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import torch.jit
from parlai.core.dict import DictionaryAgent
from parlai.core.torch_agent import TorchAgent
from parlai.utils.bpe import Gpt2BpeHelper
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
        "dict_tokenizer": "gpt2",
        "dict_lower": False,
        "dict_textfields": "text,labels",
        "dict_loaded": True,
        "bpe_debug": False,
    }

    def __init__(self, agent: TorchAgent):
        super().__init__()

        self.is_bart = agent.opt["model"] == "bart"

        # Dictionary/tokenization setup
        for key, val in self.CAIRAOKE_DICT_PARAMS.items():
            assert (
                agent.opt.get(key, val) == val
            ), f'The only currently supported value of "{key}" is {val}!'
        orig_dict: DictionaryAgent = agent.dict
        orig_bpe: Gpt2BpeHelper = orig_dict.bpe
        assert all(len(key) == 2 for key in orig_bpe.bpe_ranks.keys())
        assert not any(
            i for key in orig_bpe.bpe_ranks.keys() for i in key if "\n" in i
        ), "We need to temporarily merge the bpe_ranks dict's keys with a newline character in order to use it as a TorchScript arg, but at least one of the dict's keys contains a newline character already!"
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
        sample_tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
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

    def _get_initial_decoder_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Workaround because we can't use TGM._get_initial_decoder_input() directly.

        When we try to call that function, we get a "RuntimeError: Type 'Tuple[int,
        int]' cannot be traced. Only Tensors and (possibly nested) Lists, Dicts, and
        Tuples of Tensors can be traced" error.
        """
        bsz = x.size(0)
        return (
            torch.tensor(self.initial_decoder_input, dtype=torch.long)
            .expand(bsz, len(self.initial_decoder_input))
            .to(x.device)
        )

    def parse(self, text: str) -> List[int]:
        return self.dict.txt2vec(text)

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
        return self.dict.vec2txt(new_vec)

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
                    torch.tensor([self.start_idx], dtype=torch.long),
                    flattened_text_vec,
                    torch.tensor([self.end_idx], dtype=torch.long),
                ],
                dim=0,
            )

        # Pass through the encoder and decoder to generate tokens
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


@torch.jit.script
class ScriptableGpt2BpeHelper(object):
    """
    Version of parlai.utils.bpe.Gpt2BpeHelper that can be TorchScripted.
    """

    @classmethod
    def findall(cls, text: str) -> List[str]:
        """
        Split tokens in a manner that replicates parlai.utils.bpe.Gpt2BpeHelper.
        """
        contraction_endings = ["s", "t", "re", "ve", "m", "ll", "d"]

        tokens: List[str] = []
        idx = 0
        num_passes = 0
        while idx < len(text):
            num_passes += 1
            if num_passes > 10000:
                return ["*** Infinite loop in ScriptableGpt2BpeHelper.findall()! ***"]
            if text[idx] == "'":
                # Capture contradiction suffixes
                captured_suffix = False
                for ending in contraction_endings:
                    if text[idx + 1 : idx + 1 + len(ending)] == ending:
                        tokens.append("'" + ending)
                        idx += 1 + len(ending)
                        captured_suffix = True
                        break
                if captured_suffix:
                    continue
            if not text[idx].isspace() or (
                text[idx] == " " and idx + 1 < len(text) and not text[idx + 1].isspace()
            ):
                # Capture runs of one type of character
                if text[idx] == " ":
                    last_matching_idx = idx + 1
                else:
                    last_matching_idx = idx
                if text[last_matching_idx].isalpha():
                    while (
                        last_matching_idx + 1 < len(text)
                        and text[last_matching_idx + 1].isalpha()
                    ):
                        last_matching_idx += 1
                elif text[last_matching_idx].isnumeric():
                    while (
                        last_matching_idx + 1 < len(text)
                        and text[last_matching_idx + 1].isnumeric()
                    ):
                        last_matching_idx += 1
                else:
                    while (
                        last_matching_idx + 1 < len(text)
                        and not text[last_matching_idx + 1].isspace()
                        and not text[last_matching_idx + 1].isalpha()
                        and not text[last_matching_idx + 1].isnumeric()
                    ):
                        last_matching_idx += 1
                tokens.append(text[idx : last_matching_idx + 1])
                idx = last_matching_idx + 1
                continue
            if idx + 1 < len(text) and text[idx + 1].isspace():
                # Capture runs of space characters up until just before the final one
                last_space_idx = idx + 1
                while (
                    last_space_idx + 1 < len(text)
                    and text[last_space_idx + 1].isspace()
                ):
                    last_space_idx += 1
                if last_space_idx + 1 == len(text):
                    # Include the last char, which is a space char
                    tokens.append(text[idx : last_space_idx + 1])
                    idx = last_space_idx + 1
                else:
                    tokens.append(text[idx:last_space_idx])
                    idx = last_space_idx
                continue
            if True:
                # Capture runs of space characters
                last_space_idx = idx
                while (
                    last_space_idx + 1 < len(text)
                    and text[last_space_idx + 1].isspace()
                ):
                    last_space_idx += 1
                tokens.append(text[idx : last_space_idx + 1])
                idx = last_space_idx + 1
        return tokens

    def __init__(
        self,
        add_prefix_space: bool,
        encoder: Dict[str, str],
        byte_encoder: Dict[int, str],
        fused_key_bpe_ranks: Dict[str, float],
        special_tokens: List[str],
    ):

        self.add_prefix_space = add_prefix_space

        self.encoder = encoder
        self.decoder: Dict[str, str] = {}
        for k, v in self.encoder.items():
            self.decoder[v] = k

        self.byte_encoder = byte_encoder
        self.byte_decoder: Dict[str, int] = {}
        for k, v in self.byte_encoder.items():
            self.byte_decoder[v] = k

        self.bpe_ranks = fused_key_bpe_ranks

        # special tokens
        self._special_tokens: Dict[str, int] = {}
        for st in special_tokens:
            self._special_tokens[st] = 1

    def encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        Checks for add_prefix_space; handles accordingly.

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """
        if self.add_prefix_space:
            text = f" {text}"

        # constants for readability
        FINAL = 1
        SPLITABLE = 0
        pieces: List[Tuple[str, int]] = [(text, SPLITABLE)]

        for special_token in self._special_tokens.keys():
            i = 0
            while i < len(pieces):
                subtext, status = pieces[i]
                if status == FINAL:
                    i += 1
                    continue
                split = subtext.split(special_token)
                if len(split) > 1:
                    # special token detected, replace the chunk with small subchunks
                    # split by the special token
                    pieces.pop(i)
                    for j, piece in enumerate(split):
                        if j > 0:
                            # add the special token as a delimiter
                            pieces.insert(i + j, (special_token, FINAL))
                        pieces.insert(i + j + int(j > 0), (piece, SPLITABLE))
                else:
                    i += 1

        output: List[str] = []
        for piece, state in pieces:
            if state is FINAL:
                output.append(piece)
            else:
                output += self.helper_encode(piece)
        text = "".join(output)

        return output

    def get_pairs(self, word: List[str]) -> List[Tuple[str, str]]:
        """
        Return set of symbol pairs in a word.

        Word is represented as list of symbols (symbols being variable-length strings).

        :param word:
            word to symbolize

        :return pairs:
            set of tuples of symbols
        """
        pairs: List[Tuple[str, str]] = []
        prev_char = word[0]
        for char in word[1:]:
            pairs.append((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, word: List[str]) -> List[str]:
        """
        Convert token to BPE.

        :param word:
            list of tokens token to convert

        :return bpe_encoding:
            string bpe encoding
        """
        pairs = self.get_pairs(word)

        if len(pairs) == 0:
            return word

        while True:
            min_rank = self.bpe_ranks.get("\n".join(pairs[0]), float("inf"))
            bigram = pairs[0]
            for pair in pairs[1:]:
                current_rank = self.bpe_ranks.get("\n".join(pair), float("inf"))
                if current_rank < min_rank:
                    min_rank = current_rank
                    bigram = pair
            if "\n".join(bigram) not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                found = False
                for j in range(i, len(word)):
                    if word[j] == first:
                        new_word.extend(word[i:j])
                        i = j
                        found = True
                        break
                if not found:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word.copy()
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        return word

    def helper_encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """
        bpe_tokens: List[str] = []
        for token in self.findall(text):
            byte_encoded: List[str] = []
            for b in token:
                byte_encoded.append(self.byte_encoder[ord(b)])
            encoded: List[str] = []
            for bpe_token in self.bpe(byte_encoded):
                encoded.append(self.encoder[bpe_token])
            bpe_tokens.extend(encoded)
        return bpe_tokens

    def decode(self, tokens: List[str]) -> str:
        """
        Decode list of tokens into a text string.

        :param tokens:
            list of tokens

        :return text:
            decoded text
        """
        output: List[str] = []
        accum: List[str] = []
        for token in tokens:
            if token in self._special_tokens:
                if len(accum) > 0:
                    output.append(self.helper_decode(accum))
                    accum.clear()
                output.append(token)
            else:
                accum.append(token)
        if len(accum) > 0:
            output.append(self.helper_decode(accum))

        text = "".join(output)
        if self.add_prefix_space:
            assert text.startswith(" ")
            text = text.lstrip(" ")
        return text

    def helper_decode(self, tokens: List[str]) -> str:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens

        :return:
            decoded text
        """
        chars: List[str] = []
        for token in tokens:
            decoded_token = self.decoder[token]
            token_chars = self.utf8_chars(decoded_token)
            for char in token_chars:
                if not torch.jit.is_scripting():
                    # We iterate over "char", which is supposed to be a single
                    # character, because the TorchScripted version of the code
                    # correctly splits a string into single characters in
                    # self.utf8_chars() but the non-TorchScripted version doesn't
                    chars.extend(list(char))
                else:
                    chars.append(char)
        decoded_chars: List[str] = []
        for char in chars:
            decoded_chars.append(chr(self.byte_decoder[char]))
        return "".join(decoded_chars)

    def utf8_chars(self, s: str) -> List[str]:
        """
        An implementation of UTF8 character iteration in TorchScript. There are no
        bitwise operations in torchscript, so we compare directly to integer values.
        There isn't a lot of validation, for instance if you pass in an improperly
        encoded string with an out-of-place continuation byte, or with a non-left-to-
        right byte order, you'll get unexpected results and likely throw. Torch itself
        takes in unicode strings and encodes them as UTF8, so that should be actively
        hard to do.

        The logic is simple: looking at the current start-of-character byte.
        If its high bit is 0, it's a 1-byte character. Otherwise, the number of
        bytes is the number of leading 1s in its binary representation, so
        find that number by comparing it directly to ints with the appropriate
        representation, then append that many bytes as a character and move past
        them to the next start byte.

        From pytext.torchscript.utils.
        """
        chars: List[str] = []
        i = 0
        while i < len(s):
            byte = ord(s[i])
            if byte < 0b10000000:
                chars.append(s[i])
                i += 1
            else:
                if byte < 0b11100000:
                    num_bytes = 2
                elif byte < 0b11110000:
                    num_bytes = 3
                elif byte < 0b11111000:
                    num_bytes = 4
                elif byte < 0b11111100:
                    num_bytes = 5
                elif byte < 0b11111110:
                    num_bytes = 6
                elif byte < 0b11111111:
                    num_bytes = 7
                else:
                    num_bytes = 8
                chars.append(s[i : i + num_bytes])
                i += num_bytes
        return chars


@torch.jit.script
class ScriptableDictionaryAgent:
    """
    Builds and/or loads a dictionary.

    All code is TorchScriptable.
    """

    def __init__(
        self,
        null_token: str,
        end_token: str,
        unk_token: str,
        start_token: str,
        freq: Dict[str, int],
        tok2ind: Dict[str, int],
        ind2tok: Dict[int, str],
        bpe_add_prefix_space: bool,
        bpe_encoder: Dict[str, str],
        bpe_byte_encoder: Dict[int, str],
        fused_key_bpe_ranks: Dict[str, float],
        special_tokens: List[str],
    ):

        self.null_token = null_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.start_token = start_token

        self.freq = freq
        self.tok2ind = tok2ind
        self.ind2tok = ind2tok

        # cache unk token for later
        self._unk_token_idx = self.tok2ind[self.unk_token]

        # Initialize tokenizer
        self.bpe = ScriptableGpt2BpeHelper(
            add_prefix_space=bpe_add_prefix_space,
            encoder=bpe_encoder,
            byte_encoder=bpe_byte_encoder,
            fused_key_bpe_ranks=fused_key_bpe_ranks,
            special_tokens=special_tokens,
        )

    def _word_lookup(self, key: str) -> int:
        """
        Return index from token, or unk_token's index, or None.
        """
        if key in self.tok2ind:
            return self.tok2ind[key]
        else:
            return self._unk_token_idx

    def _index_lookup(self, key: int) -> str:
        """
        Return token from index, or unk_token.
        """
        if key in self.ind2tok:
            return self.ind2tok[key]
        else:
            return self.unk_token

    def gpt2_tokenize(self, text: str):
        """
        Tokenize using Gpt2 BPE tokenizer.
        """
        return self.bpe_tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        """
        Return a sequence of tokens from the iterable.

        Also handles special tokens for some tokenizers
        """

        # calls the selected tokenizer function e.g. 're' => re_tokenize(text)
        word_tokens = self.gpt2_tokenize(text)

        return word_tokens

    def bpe_tokenize(self, text: str) -> List[str]:
        """
        Return a sequence of BPE-tokens from the text.
        """
        return self.bpe.encode(text)

    def txt2vec(self, text: str) -> List[int]:
        """
        Convert a string to a vector (list of ints).

        First runs a sentence tokenizer, then a word tokenizer.
        """
        itr: List[int] = []
        for token in self.tokenize(str(text)):
            itr.append(self._word_lookup(token))
        return itr

    def vec2txt(self, vector: List[int]) -> str:
        """
        Convert a vector of IDs to a string.

        Converts a vector (iterable of ints) into a string, with each token separated by
        the delimiter (default ``' '``).
        """
        tokens = [self._index_lookup(idx) for idx in vector]
        text = self.bpe.decode(tokens)
        return text
