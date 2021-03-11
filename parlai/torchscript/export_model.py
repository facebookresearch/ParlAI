#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

import torch.jit
import torch.nn as nn

from parlai.core.agents import create_agent
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from parlai.core.torch_agent import TorchAgent
from parlai.torchscript.util import setup_args
from parlai.utils.bpe import Gpt2BpeHelper
from parlai.utils.io import PathManager


def export_model(opt: Opt):

    agent = create_agent(opt, requireModelExists=True)

    # Script and trace the greedy search routine
    original_module = JitGreedySearch(agent)

    inputs = opt['input'].split('|')

    print('\nGenerating given the original unscripted module:')
    for input_ in inputs:
        label = original_module(input_)
        print("LABEL: " + label)

    # # Script the module and save
    # scripted_module = torch.jit.script(original_module)
    # with PathManager.open(opt['scripted_model_file'], 'wb') as f:
    #     torch.jit.save(scripted_module, f)

    # print('\nGenerating given the scripted module:')
    # for input_ in inputs:
    #     label = scripted_module(input_)
    #     print("LABEL: " + label)


class JitGreedySearch(nn.Module):
    """
    A helper class for exporting simple greedy-search models via TorchScript.

    Models with extra inputs will need to override to include more variables.

    Utilize with:

    >>> TODO: write this
    """

    history_vecs: List[List[int]]
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
        'bpe_debug': False,
    }

    def __init__(self, agent: TorchAgent):
        super().__init__()

        self.is_bart = agent.opt['model'] == 'bart'

        # Download bpe data and json

        # Dictionary/tokenization setup
        for key, val in self.CAIRAOKE_DICT_PARAMS.items():
            assert (
                agent.opt.get(key, val) == val
            ), f'The only currently supported value of "{key}" is {val}!'
        orig_dict: DictionaryAgent = agent.dict
        orig_bpe: Gpt2BpeHelper = orig_dict.bpe
        assert all(len(key) == 2 for key in orig_bpe.bpe_ranks.keys())
        assert not any(
            i for key in orig_bpe.bpe_ranks.keys() for i in key if '\n' in i
        ), "We need to temporarily merge the bpe_ranks dict's keys with a newline character in order to use it as a TorchScript arg, but at least one of the dict's keys contains a newline character already!"
        fused_key_bpe_ranks = {
            '\n'.join(key): float(val) for key, val in orig_bpe.bpe_ranks.items()
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
            bpe_add_prefix_space=agent.opt['bpe_add_prefix_space'],
            bpe_encoder=orig_bpe.encoder,
            bpe_byte_encoder=orig_bpe.byte_encoder,
            fused_key_bpe_ranks=fused_key_bpe_ranks,
        )

        # History tracking and start/end tokens
        self.history_vecs = []
        self.delimiter_tok = agent.history.delimiter_tok
        self.history_size = agent.opt['history_size']
        if agent.opt.get('history_add_global_end_token', None) is not None:
            self.global_end_token = agent.dict[agent.dict.end_token]
        else:
            self.global_end_token = None
        self.text_truncate = agent.opt.get('text_truncate') or agent.opt['truncate']
        self.text_truncate = self.text_truncate if self.text_truncate >= 0 else None

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

    def parse(self, text: str) -> List[int]:
        return self.dict.txt2vec(text)

    def _update_vecs(self, text: str):
        if self.history_size > 0:
            while len(self.history_vecs) >= self.history_size:
                self.history_vecs.pop(0)
        self.history_vecs.append(self.parse(text))

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

    def forward(self, input_: str, max_len: int = 128) -> str:
        # TODO: docstring

        # Vectorize this line of context
        print(" TEXT: " + input_)
        self._update_vecs(input_)

        # Get full history vec
        text_vecs: List[List[int]] = []
        for vec in self.history_vecs[:-1]:
            text_vecs += [vec]
            text_vecs += [self.delimiter_tok]
        text_vecs += [self.history_vecs[-1]]
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
        label = self._v2t(generations[0].tolist())
        self._update_vecs(label)

        return label


# @torch.jit.script
class ScriptableGpt2BpeHelper(object):
    """
    Version of parlai.utils.bpe.Gpt2BpeHelper that can be TorchScripted.
    """

    # DEFAULT_ENCODER_JSON = (
    #     'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
    # )
    # DEFAULT_VOCAB_BPE = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
    ERRORS_METHOD = 'replace'

    @classmethod
    def findall(cls, text: str) -> List[str]:
        """
        Split tokens in a manner that approximates parlai.utils.bpe.Gpt2BpeHelper.
        """
        contraction_endings = ['s', 't', 're', 've', 'm', 'll', 'd']

        tokens: List[str] = []
        idx = 0
        while idx < len(text):
            if text[idx] == "'":
                # Capture contradiction suffixes
                for ending in contraction_endings:
                    if text[idx + 1 : idx + 1 + len(ending)] == ending:
                        tokens.append("'" + ending)
                        idx += 1 + len(ending)
                        break
                continue
            if not text[idx].isspace() or (
                text[idx] == ' ' and idx + 1 < len(text) and not text[idx + 1].isspace()
            ):
                # Capture runs of one type of character
                if text[idx] == ' ':
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
    ):
        """
        Override init to build the data.
        """
        self.add_prefix_space = add_prefix_space

        #     super().__init__(opt, shared)
        #     if self.lower:
        #         warn_once('Are you sure you want to lower case your BPE dictionary?')
        #
        #     if self.maxtokens > 0 or self.minfreq > 0:
        #         raise ValueError(
        #             'You should not filter vocabulary with using --dict-tokenizer bytelevelbpe'
        #             ' (no --dict-minfreq or --dict-maxtokens).'
        #         )

        # self.bpe_data, self.json_path, self.merge_path = self._build_data()

        # build encoder & decoder
        self.encoder = encoder

        self.decoder: Dict[str, str] = {}
        for k, v in self.encoder.items():
            self.decoder[v] = k

        # bpe_merges = [
        #     tuple(merge_str.split()) for merge_str in self.bpe_data.split('\n')[1:-1]
        # ]
        self.byte_encoder = byte_encoder
        self.byte_decoder: Dict[str, int] = {}
        for k, v in self.byte_encoder.items():
            self.byte_decoder[v] = k
        self.bpe_ranks = fused_key_bpe_ranks

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        # self.pat = regex.compile(
        #     r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # )

    def encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        Checks for add_prefix_space; handles accordingly

        NOTE: DO NOT OVERRIDE

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """
        if self.add_prefix_space:
            text = f' {text}'
        return self.helper_encode(text)

    # def _build_data(self) -> Tuple[str, str, str]:
    #     """
    #     Build data.
    #
    #     Maybe download the appropriate data.
    #
    #     :return (bpe_data, json_path):
    #         bpe_data and path to encoder json
    #     """
    #     data_path = os.path.join(self.opt['datapath'], 'gpt2')
    #     vocab_path = os.path.join(data_path, 'vocab.bpe')
    #     json_path = os.path.join(data_path, 'encoder.json')
    #     if not PathManager.exists(vocab_path) or not PathManager.exists(json_path):
    #         make_dir(data_path)
    #         download(self.DEFAULT_VOCAB_BPE, data_path, 'vocab.bpe')
    #         download(self.DEFAULT_ENCODER_JSON, data_path, 'encoder.json')
    #     with PathManager.open(vocab_path, 'r', encoding="utf-8") as f:
    #         bpe_data = f.read()
    #
    #     return bpe_data, json_path, vocab_path

    # def _build_encoder(self, json_path: str) -> Dict[str, str]:
    #     """
    #     Build and return the encoder.
    #
    #     :param json_path:
    #         path to encoder json file
    #
    #     :return:
    #         encoder, mapping tokens to unicode reps
    #     """
    #     with PathManager.open(json_path, 'r', encoding='utf8') as f:
    #         encoder = json.load(f)
    #     for each_token in encoder.keys():
    #         new_token = ''.join(
    #             # escape nonprintable characters
    #             '\\' + hex(b).lstrip('0') if (b > 127 or b < 32) else chr(b)
    #             for b in each_token.encode('utf-8')
    #         )
    #         encoder[each_token] = new_token
    #
    #     return encoder

    # @lru_cache()
    # def bytes_to_unicode(self) -> Dict[int, str]:
    #     """
    #     Returns list of utf-8 byte and a corresponding list of unicode strings.
    #
    #     The reversible bpe codes work on unicode strings. This means you need a large #
    #     of unicode characters in your vocab if you want to avoid UNKs. When you're at
    #     something like a 10B token dataset you end up needing around 5K for decent
    #     coverage. This is a signficant percentage of your normal, say, 32K bpe vocab. To
    #     avoid that, we want lookup tables between utf-8 bytes and unicode strings. And
    #     avoids mapping to whitespace/control characters the bpe code barfs on.
    #     """
    #     bs: List[int] = (
    #         list(range(ord("!"), ord("~") + 1))
    #         + list(range(ord("¡"), ord("¬") + 1))
    #         + list(range(ord("®"), ord("ÿ") + 1))
    #     )
    #     cs: List[int] = bs[:]
    #     n = 0
    #     for b in range(2 ** 8):
    #         if b not in bs:
    #             bs.append(b)
    #             cs.append(2 ** 8 + n)
    #             n += 1
    #     str_cs: List[str] = [chr(n) for n in cs]
    #     return dict(zip(bs, str_cs))

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

    def bpe(self, token: str) -> str:
        """
        Convert token to BPE.

        :param token:
            token to convert

        :return bpe_encoding:
            string bpe encoding
        """
        word = list(token)
        pairs = self.get_pairs(word)

        if len(pairs) == 0:
            return token

        while True:
            min_rank = self.bpe_ranks.get('\n'.join(pairs[0]), float('inf'))
            bigram = pairs[0]
            for pair in pairs[1:]:
                current_rank = self.bpe_ranks.get('\n'.join(pair), float('inf'))
                if current_rank < min_rank:
                    min_rank = current_rank
                    bigram = pair
            if '\n'.join(bigram) not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                for j in range(i, len(word)):
                    if word[j] == first:
                        new_word.extend(word[i:j])
                        i = j
                        break
                else:
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
        return ' '.join(word)

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
            token = ''.join(byte_encoded)
            encoded: List[str] = []
            for bpe_token in self.bpe(token).split(' '):
                encoded.append(self.encoder[bpe_token])
            bpe_tokens.extend(encoded)
        return bpe_tokens

    def decode(self, tokens: List[str]) -> str:
        """
        Decode list of tokens into a text string.

        NOTE: DO NOT OVERRIDE

        :param tokens:
            list of tokens

        :return text:
            decoded text
        """
        # no special tokens found, we can fall back
        text = self.helper_decode(tokens)
        if self.add_prefix_space:
            assert text.startswith(' ')
            text = text.lstrip(' ')
        return text

    def helper_decode(self, tokens: List[str]) -> str:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens

        :return text:
            decoded text
        """
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors=self.ERRORS_METHOD
        )
        return text

    # def sync_with_dict(self, dict_agent: ScriptableDictionaryAgent):
    #     """
    #     Sync with dictionary agent.
    #
    #     Just add all of the tokens to the dict
    #
    #     NOTE: How does this handle special tokens?
    #
    #     :param dict_agent:
    #         A DictionaryAgent instantiation
    #     """
    #     for each_token in self.encoder.values():
    #         dict_agent.add_token(each_token)
    #         dict_agent.freq[each_token] = 1

    # def save(self, dir_name: str, file_name: str):
    #     """
    #     Save appropriate files.
    #
    #     :param dir_name:
    #         directory to save.
    #     :param file_name:
    #         file to save.
    #     """
    #     out_json_path = os.path.join(dir_name, file_name + "-vocab.json")
    #     out_merge_path = os.path.join(dir_name, file_name + "-merges.txt")
    #     # Possibly bad assumption: if the destination file already exists,
    #     # we don't need to copy it over again.
    #     if not PathManager.exists(out_json_path):
    #         logging.info(f"Copying {self.json_path} to {out_json_path}")
    #         PathManager.copy(self.json_path, out_json_path)
    #     if not PathManager.exists(out_merge_path):
    #         logging.info(f"Copying {self.merge_path} to {out_merge_path}")
    #         PathManager.copy(self.merge_path, out_merge_path)


# @torch.jit.script
class ScriptableDictionaryAgent:
    """
    Builds and/or loads a dictionary. All code is TorchScriptable.
    """

    # default_lang = 'english'
    # default_maxngram = -1
    # default_minfreq = 0
    # default_maxtokens = -1
    # default_null = '__null__'
    # default_start = '__start__'
    # default_end = '__end__'
    # default_unk = '__unk__'
    # default_tok = 're'
    # default_lower = False
    # default_textfields = 'text,labels'
    #
    # @classmethod
    # def add_cmdline_args(
    #     cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    # ) -> ParlaiParser:
    #     """
    #     Add commandline arguments related to the dictionary.
    #     """
    #     dictionary = parser.add_argument_group('Dictionary Arguments')
    #     dictionary.add_argument(
    #         '-df',
    #         '--dict-file',
    #         help='path to dictionary file. defaults to [model_file].dict if '
    #         'not set and model_file is set.',
    #         hidden=True,
    #     )
    #     dictionary.add_argument(
    #         '--dict-initpath',
    #         hidden=True,
    #         help='path to a saved dictionary to load tokens / counts from to '
    #         'seed the dictionary with initial tokens and/or frequencies',
    #     )
    #     dictionary.add_argument(
    #         '--dict-language',
    #         default=DictionaryAgent.default_lang,
    #         hidden=True,
    #         help='sets language for the punkt sentence tokenizer',
    #     )
    #     dictionary.add_argument(
    #         '--dict-max-ngram-size',
    #         type=int,
    #         hidden=True,
    #         default=DictionaryAgent.default_maxngram,
    #         help='looks for ngrams of up to this size. this is ignored when '
    #         'building the dictionary. note: this takes approximate '
    #         'runtime of len(sentence)^max_ngram_size',
    #     )
    #     dictionary.add_argument(
    #         '--dict-minfreq',
    #         default=DictionaryAgent.default_minfreq,
    #         type=int,
    #         help='minimum frequency of words to include them in sorted '
    #         'dict or minimum frequency of bpe codecs',
    #         hidden=True,
    #     )
    #     dictionary.add_argument(
    #         '--dict-maxtokens',
    #         default=DictionaryAgent.default_maxtokens,
    #         type=int,
    #         help='max number of tokens to include in dictionary or bpe codecs',
    #         hidden=True,
    #     )
    #     dictionary.add_argument(
    #         '--dict-nulltoken',
    #         default=DictionaryAgent.default_null,
    #         hidden=True,
    #         help='empty token, can be used for padding or just empty values',
    #     )
    #     dictionary.add_argument(
    #         '--dict-starttoken',
    #         default=DictionaryAgent.default_start,
    #         hidden=True,
    #         help='token for starting sentence generation, if needed',
    #     )
    #     dictionary.add_argument(
    #         '--dict-endtoken',
    #         default=DictionaryAgent.default_end,
    #         hidden=True,
    #         help='token for end of sentence markers, if needed',
    #     )
    #     dictionary.add_argument(
    #         '--dict-unktoken',
    #         default=DictionaryAgent.default_unk,
    #         hidden=True,
    #         help='token to return for unavailable words',
    #     )
    #     dictionary.add_argument(
    #         '-tok',
    #         '--dict-tokenizer',
    #         default=DictionaryAgent.default_tok,
    #         help='Which tokenizer to use. Defaults to "split", which splits '
    #         'on whitespace as well as recognizing basic punctuation. '
    #         'Other options include nltk, gpt2 and bytelevelbpe.',
    #         hidden=True,
    #     )
    #     dictionary.add_argument(
    #         '--dict-lower',
    #         default=DictionaryAgent.default_lower,
    #         type='bool',
    #         help='Whether or not to lowercase all text seen.',
    #         hidden=True,
    #     )
    #     dictionary.add_argument(
    #         '--bpe-debug',
    #         action='store_true',
    #         hidden=True,
    #         help='Leave BPE tokens untouched in output. Useful for debugging.',
    #     )
    #     dictionary.add_argument(
    #         '--dict-textfields',
    #         default=DictionaryAgent.default_textfields,
    #         hidden=True,
    #         help='Observation fields which dictionary learns vocabulary from. '
    #         'Tasks with additional fields may add to this list to handle '
    #         'any extra vocabulary.',
    #     )
    #     dictionary = BPEHelper.add_cmdline_args(dictionary, partial_opt=partial_opt)
    #     return dictionary

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
    ):
        """
        Initialize DictionaryAgent.
        """
        # self.opt = copy.deepcopy(opt)
        # self.minfreq = opt.get('dict_minfreq', DictionaryAgent.default_minfreq)
        self.null_token = null_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.start_token = start_token
        # self.max_ngram_size = opt.get(
        #     'dict_max_ngram_size', DictionaryAgent.default_maxngram
        # )
        # self.tokenizer = opt.get('dict_tokenizer', DictionaryAgent.default_tok)
        # self.lower = opt.get('dict_lower', DictionaryAgent.default_lower)
        # self.maxtokens = opt.get('dict_maxtokens', DictionaryAgent.default_maxtokens)
        # self.textfields = opt.get(
        #     'dict_textfields', DictionaryAgent.default_textfields
        # ).split(",")
        #
        # # used to signal whether we should use training time tricks, like bpe droput
        # self._tokenization_mode = TokenizationMode.TEST_TIME_LABEL
        #
        # try:
        #     self.tokenizer_fun = getattr(self, self.tokenizer + '_tokenize')
        # except AttributeError:
        #     raise AttributeError(
        #         'tokenizer type {} not yet supported'.format(self.tokenizer)
        #     )
        #
        # if shared:
        #     self.freq = shared.get('freq', {})
        #     self.tok2ind = shared.get('tok2ind', {})
        #     self.ind2tok = shared.get('ind2tok', {})
        # else:
        #     self.additional_special_tokens: List[str] = []
        self.freq = freq
        self.tok2ind = tok2ind
        self.ind2tok = ind2tok

        # if self.null_token:
        #     self.add_token(self.null_token)
        #
        # if self.start_token:
        #     # set special start of sentence word token
        #     self.add_token(self.start_token)
        #
        # if self.end_token:
        #     # set special end of sentence word token
        #     self.add_token(self.end_token)
        #
        # if self.unk_token:
        #     # set special unknown word token
        #     self.add_token(self.unk_token)

        #     loaded = False
        #     # If data built via pytorch data teacher, we need to load prebuilt dict
        #     if opt.get('dict_file'):
        #         opt['dict_file'] = modelzoo_path(opt.get('datapath'), opt['dict_file'])
        #         if PathManager.exists(opt['dict_file']):
        #             # load pre-existing dictionary
        #             self.load(opt['dict_file'])
        #             loaded = True
        #
        #     if not loaded and opt.get('dict_initpath'):
        #         # load seed dictionary
        #         opt['dict_initpath'] = modelzoo_path(
        #             opt.get('datapath'), opt['dict_initpath']
        #         )
        #         # don't check isfile first, should fail if file not found
        #         self.load(opt['dict_initpath'])
        #     opt['dict_loaded'] = loaded

        # cache unk token for later
        self._unk_token_idx = self.tok2ind[self.unk_token]

        # # initialize tokenizers
        # if self.tokenizer == 'nltk':
        #     try:
        #         import nltk
        #     except ImportError:
        #         raise ImportError('Please install nltk (pip install nltk)')
        #     # nltk-specific setup
        #     st_path = 'tokenizers/punkt/{0}.pickle'.format(opt['dict_language'])
        #     try:
        #         self.sent_tok = nltk.data.load(st_path)
        #     except LookupError:
        #         nltk.download('punkt')
        #         self.sent_tok = nltk.data.load(st_path)
        #     self.word_tok = nltk.tokenize.treebank.TreebankWordTokenizer()
        # elif self.tokenizer in ['bpe', 'gpt2', 'bytelevelbpe', 'slow_bytelevel_bpe']:
        self.bpe = ScriptableGpt2BpeHelper(
            add_prefix_space=bpe_add_prefix_space,
            encoder=bpe_encoder,
            byte_encoder=bpe_byte_encoder,
            fused_key_bpe_ranks=fused_key_bpe_ranks,
        )
        # self.bpe.sync_with_dict(self)

        # # if not shared:
        # if self.null_token:
        #     # fix count for null token to one billion and three
        #     self.freq[self.null_token] = 1000000003
        #
        # if self.start_token:
        #     # fix count for start of sentence token to one billion and two
        #     self.freq[self.start_token] = 1000000002
        #
        # if self.end_token:
        #     # fix count for end of sentence token to one billion and one
        #     self.freq[self.end_token] = 1000000001
        #
        # if self.unk_token:
        #     # fix count for unknown token to one billion
        #     self.freq[self.unk_token] = 1000000000
        #
        #     if opt.get('dict_file'):
        #         self.save_path = opt['dict_file']

    # def is_prebuilt(self):
    #     """
    #     Indicates whether the dictionary is fixed, and does not require building.
    #     """
    #     return self.tokenizer == 'gpt2'

    # def add_token(self, word):
    #     """
    #     Add a single token to the dictionary.
    #     """
    #     if word not in self.tok2ind:
    #         index = len(self.tok2ind)
    #         self.tok2ind[word] = index
    #         self.ind2tok[index] = word

    # def __contains__(self, key):
    #     """
    #     Return if the dictionary contains the key.
    #
    #     If key is an int, returns whether the key is in the indices. If key is a str,
    #     return if the token is in the dict of tokens.
    #     """
    #     if type(key) == int:
    #         return key in self.ind2tok
    #     elif type(key) == str:
    #         return key in self.tok2ind

    def _word_lookup(self, key: str) -> int:
        # return index from token, or unk_token's index, or None
        if key in self.tok2ind:
            return self.tok2ind[key]
        else:
            return self._unk_token_idx

    def _index_lookup(self, key: int) -> str:
        # return token from index, or unk_token
        if key in self.ind2tok:
            return self.ind2tok[key]
        else:
            return self.unk_token

    # def __getitem__(self, key):
    #     """
    #     Lookup the word or ID.
    #
    #     If key is an int, returns the corresponding token. If it does not exist, return
    #     the unknown token. If key is a str, return the token's index. If the token is
    #     not in the dictionary, return the index of the unknown token. If there is no
    #     unknown token, return ``None``.
    #     """
    #     if type(key) == str:
    #         return self._word_lookup(key)
    #     if type(key) == int:
    #         return self._index_lookup(key)
    #
    # def __len__(self):
    #     return len(self.tok2ind)
    #
    # def __setitem__(self, key, value):
    #     """
    #     Set the frequency for a word to a value.
    #
    #     If the key is not in the dictionary, add it to the dictionary and set its
    #     frequency to value.
    #     """
    #     key = str(key)
    #     if self.lower:
    #         key = key.lower()
    #     self.freq[key] = int(value)
    #     self.add_token(key)
    #
    # def keys(self):
    #     """
    #     Return all the words in the dictionary.
    #     """
    #     return self.tok2ind.keys()
    #
    # def nltk_tokenize(self, text, building=False):
    #     """
    #     Tokenize using NLTK PunktTokenizer.
    #
    #     Uses nltk-trained PunktTokenizer for sentence tokenization and Treebank Word
    #     Tokenizer for tokenizing words within sentences.
    #     """
    #     return (
    #         token
    #         for sent in self.sent_tok.tokenize(text)
    #         for token in self.word_tok.tokenize(sent)
    #     )

    def gpt2_tokenize(self, text: str):
        """
        Tokenize using Gpt2 BPE tokenizer.
        """
        return self.bpe_tokenize(text)

    # def slow_bytelevel_bpe_tokenize(self, text):
    #     """
    #     Tokenize using Gpt2 BPE tokenizer.
    #     """
    #     return self.bpe_tokenize(text)
    #
    # def bytelevelbpe_tokenize(self, text):
    #     """
    #     Tokenize using Gpt2 BPE tokenizer.
    #     """
    #     return self.bpe_tokenize(text)
    #
    # @staticmethod
    # def re_tokenize(text):
    #     r"""
    #     Tokenize using a liberal regular expression.
    #
    #     Find boundaries between word characters, newlines, and non-word
    #     non-whitespace tokens ``(r'[\\w\\n]+ | [^\\w\\s] | \\n')``.
    #
    #     This splits along whitespace and punctuation and keeps the newline as
    #     a token in the returned list.
    #     """
    #     return RETOK.findall(text)
    #
    # @staticmethod
    # def split_tokenize(text):
    #     """
    #     Tokenize on whitespace and some limited punctuation.
    #
    #     Splits tokens based on whitespace after adding whitespace around
    #     punctuation.
    #
    #     Use re_tokenize if you want more robust handling of punctuation.
    #     """
    #     return (
    #         text.replace('.', ' . ')
    #         .replace(',', ' , ')
    #         .replace(';', ' ; ')
    #         .replace(':', ' : ')
    #         .replace('!', ' ! ')
    #         .replace('?', ' ? ')
    #         .split()
    #     )
    #
    # @staticmethod
    # def space_tokenize(text):
    #     """
    #     Tokenize exactly on spaces.
    #
    #     Useful when text is pre-tokenized.
    #     """
    #     return text.strip().split(' ')
    #
    # def span_tokenize(self, text):
    #     """
    #     Tokenize and find  starting index of each token in the original string.
    #     """
    #     tokens = self.tokenize(text)
    #     curr_idx = 0
    #     indices = []
    #     for t in tokens:
    #         while text[curr_idx] != t[0]:
    #             curr_idx += 1
    #         indices.append((curr_idx, curr_idx + len(t)))
    #         curr_idx += len(t)
    #     return tokens, indices

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

    # def add_to_dict(self, tokens):
    #     """
    #     Build dictionary from the list of provided tokens.
    #     """
    #     self.built = False
    #     for token in tokens:
    #         self.add_token(token)
    #         self.freq[token] += 1
    #
    # def remove_tail(self, min_freq):
    #     """
    #     Remove elements below the frequency cutoff from the dictionary.
    #     """
    #     to_remove = []
    #     for token, freq in self.freq.items():
    #         if freq < min_freq:
    #             # queue up removals since can't mutate dict during iteration
    #             to_remove.append(token)
    #
    #     for token in to_remove:
    #         del self.freq[token]
    #         idx = self.tok2ind.pop(token)
    #         del self.ind2tok[idx]
    #
    # def _remove_non_bpe(self):
    #     """
    #     Set the dictionary vocab to the bpe vocab, merging counts.
    #     """
    #     to_remove = []
    #     to_add = []
    #     for token, freq in self.freq.items():
    #         tokens = self.bpe_tokenize(token)
    #         if len(tokens) != 1:
    #             for t in tokens:
    #                 to_add.append((t, freq))
    #             to_remove.append(token)
    #     for token in to_remove:
    #         del self.freq[token]
    #         idx = self.tok2ind.pop(token)
    #         del self.ind2tok[idx]
    #     for token, freq in to_add:
    #         self.add_token(token)
    #         self.freq[token] += freq
    #
    # def resize_to_max(self, maxtokens):
    #     """
    #     Trims the dictionary to the maximum number of tokens.
    #     """
    #     if maxtokens >= 0 and len(self.tok2ind) > maxtokens:
    #         for k in range(maxtokens, len(self.ind2tok)):
    #             v = self.ind2tok[k]
    #             del self.ind2tok[k]
    #             del self.tok2ind[v]
    #             del self.freq[v]
    #
    # def load(self, filename):
    #     """
    #     Load pre-existing dictionary in 'token[<TAB>count]' format.
    #
    #     Initialize counts from other dictionary, or 0 if they aren't included.
    #     """
    #     logging.info(f'loading dictionary from {filename}')
    #
    #     lower_special = self.null_token == self.null_token.lower()
    #     SPECIAL_TOKENS = {'__UNK__', '__NULL__', '__END__', '__START__'}
    #     with PathManager.open(filename, 'r', encoding='utf-8', errors='ignore') as read:
    #         for line in read:
    #             split = line.strip().split('\t')
    #             token = unescape(split[0])
    #             if lower_special and token in SPECIAL_TOKENS:
    #                 token = token.lower()
    #             cnt = int(split[1]) if len(split) > 1 else 0
    #             self.freq[token] = cnt
    #             self.add_token(token)
    #     logging.info(f'num words = {len(self)}')
    #
    # def save(self, filename=None, append=False, sort=True):
    #     """
    #     Save dictionary to file.
    #
    #     Format is 'token<TAB>count' for every token in the dictionary, sorted
    #     by count with the most frequent words first.
    #
    #     If ``append`` (default ``False``) is set to ``True``, appends instead of
    #     overwriting.
    #
    #     If ``sort`` (default ``True``), then first sort the dictionary before saving.
    #     """
    #     filename = self.opt['dict_file'] if filename is None else filename
    #     make_dir(os.path.dirname(filename))
    #
    #     if self.tokenizer in ['bpe', 'gpt2', 'bytelevelbpe', 'slow_bytelevel_bpe']:
    #         needs_removal = self.bpe.finalize(
    #             self.freq, num_symbols=self.maxtokens, minfreq=self.minfreq
    #         )
    #         if needs_removal:
    #             self._remove_non_bpe()
    #         elif filename != self.opt.get('dict_file'):
    #             # need to copy over the old codecs file
    #             self.bpe.copy_codecs_file(filename + '.codecs')
    #         if sort and self.bpe.should_sort():
    #             self.sort(trim=False)
    #     elif sort:
    #         self.sort(trim=True)
    #
    #     logging.info(f'Saving dictionary to {filename}')
    #
    #     mode = 'a' if append else 'w'
    #     with PathManager.open(filename, mode, encoding='utf-8') as write:
    #         for i in self.ind2tok.keys():
    #             tok = self.ind2tok[i]
    #             cnt = self.freq[tok]
    #             write.write('{tok}\t{cnt}\n'.format(tok=escape(tok), cnt=cnt))
    #
    #     # save opt file
    #     with PathManager.open(filename + '.opt', 'w', encoding='utf-8') as handle:
    #         json.dump(self.opt, handle, indent=4)
    #     # save the byte level bpe model file as well
    #     if self.tokenizer == 'bytelevelbpe' or self.tokenizer == 'slow_bytelevel_bpe':
    #         # This saves filename-vocab.json and filename-merges.txt as
    #         # hugging face tokenizer does
    #         self.bpe.save(os.path.dirname(filename), os.path.basename(filename))
    #
    # def sort(self, trim=True):
    #     """
    #     Sort the dictionary.
    #
    #     Inline operation. Rearranges the dictionary so that the elements with
    #     the lowest index have the highest counts. This reindexes the dictionary
    #     according to the sorted frequencies, breaking ties alphabetically by
    #     token.
    #
    #     :param bool trim:
    #         If True, truncate the dictionary based on minfreq and maxtokens.
    #     """
    #     if trim and self.tokenizer == 'gpt2':
    #         raise RuntimeError("You should not trim the dictionary when using gpt-2.")
    #     if trim and self.tokenizer == 'bytelevelbpe':
    #         raise RuntimeError(
    #             "You should not trim the dictionary when using bytelevelbpe."
    #         )
    #     # sort first by count, then alphabetically
    #     if trim:
    #         self.remove_tail(self.minfreq)
    #     sorted_pairs = sorted(self.freq.items(), key=lambda x: (-x[1], x[0]))
    #     new_tok2ind = {}
    #     new_ind2tok = {}
    #     for i, (tok, _) in enumerate(sorted_pairs):
    #         new_tok2ind[tok] = i
    #         new_ind2tok[i] = tok
    #     self.tok2ind = new_tok2ind
    #     self.ind2tok = new_ind2tok
    #     if trim:
    #         self.resize_to_max(self.maxtokens)
    #     assert len(self.freq) == len(self.ind2tok) == len(self.tok2ind)
    #     return sorted_pairs
    #
    # def parse(self, txt_or_vec, vec_type=list):
    #     """
    #     Parse either text or a vector of indices.
    #
    #     Calls `~txt2vec` if `txt_or_vec is a string, or `~vec2txt` otherwise.
    #
    #     :param vec_type:
    #         type of the returned vector if the input is a string.
    #     """
    #     # TODO: try to deprecate this, preferring straight txt2vec
    #     if type(txt_or_vec) == str:
    #         return self.txt2vec(txt_or_vec, vec_type)
    #     else:
    #         return self.vec2txt(txt_or_vec)

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

    # def act(self):
    #     """
    #     Add words in the last observation to the dictionary.
    #
    #     This checks any fields in the message present in the --dict-textfields argument
    #     (e.g. "text,labels").
    #     """
    #     for textfield in self.textfields:
    #         source = self.observation.get(textfield)
    #         if source is None:
    #             continue
    #         # fields may be singleton strings or lists of strings.
    #         # wrap the singleton strings in a list to iterate over them
    #         if type(source) is str:
    #             source = [source]
    #         for text in source:
    #             if text:
    #                 self.add_to_dict(self.tokenize(text))
    #     return {'id': 'Dictionary'}
    #
    # def share(self):
    #     """
    #     Share internal dicts.
    #     """
    #     shared = super().share()
    #     shared['freq'] = self.freq
    #     shared['tok2ind'] = self.tok2ind
    #     shared['ind2tok'] = self.ind2tok
    #     return shared
    #
    # def shutdown(self):
    #     """
    #     Save on shutdown if ``save_path`` is set.
    #     """
    #     if hasattr(self, 'save_path'):
    #         self.save(self.save_path)
    #
    # def __str__(self):
    #     """
    #     Return string representation of frequencies in dictionary.
    #     """
    #     return str(self.freq)
    #
    # def set_tokenization_mode(self, mode: TokenizationMode):
    #     """
    #     Indicate what "kind" of tokenization is being done.
    #
    #     This can be Training Time / Testing Time, and it can be over
    #     context or labels.
    #
    #     This is used to signal from TorchAgent to the dict that it's allowed
    #     to enable things like BPE dropout. It is NOT used to indicate whether
    #     the dictionary itself is in training time.
    #
    #     Use True for training time, False for not.
    #     """
    #     self._context_mode = mode
    #     if hasattr(self, 'bpe'):
    #         # enable bpe dropout only in texts at training time. disable all
    #         # other times
    #         self.bpe.enable_bpe_dropout(mode == TokenizationMode.TRAIN_TIME_TEXT)


if __name__ == '__main__':
    parser_ = setup_args()
    opt_ = parser_.parse_args()
    export_model(opt_)
