#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
A script for converting a fairseq model to ParlAI model.

Specifically works for transformer-based models.

Example Usage:
python parlai/agents/bart/convert_fairseq_to_parlai.py \
--input /path/to/checkpoint_best.pt \
--merge /path/to/bpe-merges.txt \
--vocab /path/to/bpe-vocab.json \
--output ./model.test --space True --activation gelu
"""

from collections import OrderedDict
import os
import torch
from torch.serialization import default_restore_location
from typing import Any, Dict, List

from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
from parlai.utils.io import PathManager
from parlai.scripts.vacuum import Vacuum


TRANSFORMER_PARAMETER_MAPPING = {
    'attention_heads': 'n_heads',
    'embed_dim': 'embedding_size',
    'ffn_embed_dim': 'ffn_size',
    'learned_pos': 'learn_positional_embeddings',
}
TRANSFORMER_DROPOUT = {'dropout', 'attention_dropout'}
EMBEDDING_DICT_MAPPING = {
    'embed_tokens': 'embeddings',
    'embed_positions': 'position_embeddings',
}
FFN_MAPPING = {
    'fc1': 'ffn.lin1',
    'fc2': 'ffn.lin2',
    'layer_norms.0': 'norm1',
    'layer_norms.1': 'norm2',
    'out_proj': 'out_lin',
}


class ConversionScript(ParlaiScript):
    """
    Script to convert a transformer-based fairseq model to ParlAI Model.
    """

    @classmethod
    def setup_args(cls) -> ParlaiParser:
        parser = ParlaiParser()
        parser.add_argument(
            '--input',
            type=str,
            nargs='+',
            help='The input fairseq model path. Specify multiple to imply a join is necessary',
        )
        parser.add_argument('--output', type=str, help='The output ParlAI model path')
        parser.add_argument(
            '--vocab', type=str, help='The hugging face vocab file path, if applicable'
        )
        parser.add_argument(
            '--merge', type=str, help='The hugging face merge file path, if applicable'
        )
        parser.add_argument(
            '--add-prefix-space',
            type='bool',
            default=True,
            help='Add prefix space for hugging face bpe',
        )
        parser.add_argument(
            '--activation',
            type=str,
            help='Activation function',
            choices=['relu', 'gelu'],
            default='gelu',
        )
        parser.add_argument(
            '--tokenizer',
            type=str,
            help='Dict tokenizer',
            choices=['bytelevelbpe', 'gpt2'],
            default='bytelevelbpe',
        )
        parser.add_argument('--delimiter', type=str, default='  ', help='Delimiter')
        parser.add_argument(
            '--retain-bos-emb',
            type='bool',
            default=False,
            help='Retain the BOS embedding.',
        )
        parser.add_argument(
            '--model',
            type=str,
            default='transformer/generator',
            help='Which ParlAI agent to use.',
        )
        parser.add_argument(
            '--fp16', type='bool', default=False, help='Whether to initialize with fp16'
        )
        parser.add_argument(
            '--history-add-global-end-token',
            type='nonestr',
            default='end',
            hidden=True,
            choices=[None, 'end'],
            help='Add special token to the end of history encoding.',
        )
        return parser

    def run(self):
        """
        Run Conversion.

        Print sample agent act after conversion.
        """
        # 1. load state
        self.state = self.load_fairseq_checkpoint()
        # 2. get parlai options
        opt = self.get_parlai_opt()
        # 3. create agent and convert model weights
        self.agent = create_agent(opt)
        converted = self.convert_model_weight(opt)
        self.agent.model.load_state_dict(converted, True)
        self.agent.opt.pop('converting', None)
        self.agent.save(self.opt['output'])
        # kill the optimizer
        Vacuum.main(model_file=self.opt['output'], no_backup=True)
        # 4. enjoy!
        self.print_agent_act()

    def get_parlai_opt(self) -> Opt:
        """
        Parser for converting fairseq argument to ParlAI opt.

        :return opt:
            opt parsed by ParlAI Parser
        """
        # assume encoder/decoder symmetrical except for number of layers
        state = self.state
        fairseq_args = state['args'].__dict__

        transformer_common_config = {}

        # 1. Map transformer params
        for each in TRANSFORMER_PARAMETER_MAPPING:
            transformer_common_config[
                TRANSFORMER_PARAMETER_MAPPING[each]
            ] = fairseq_args[f'encoder_{each}']
        # 2. Map dropout
        for each in TRANSFORMER_DROPOUT:
            transformer_common_config[each] = fairseq_args[each]

        if 'activation_dropout' in fairseq_args:
            transformer_common_config['relu_dropout'] = fairseq_args[
                'activation_dropout'
            ]
        else:
            transformer_common_config['relu_dropout'] = fairseq_args['relu_dropout']

        # 3. Map other options
        transformer_common_config.update(
            {
                'model': self.opt['model'],
                # number of layers
                'n_encoder_layers': fairseq_args['encoder_layers'],
                'n_decoder_layers': fairseq_args['decoder_layers'],
                # tokenization args
                'dict_tokenizer': self.opt['tokenizer'],
                'bpe_vocab': self.opt['vocab'],
                'bpe_merge': self.opt['merge'],
                'n_positions': fairseq_args['max_source_positions'],
            }
        )

        # 4. Embedding scale
        if 'encoder_embed_scale' in fairseq_args:
            transformer_common_config['embeddings_scale'] = (
                fairseq_args['encoder_embed_scale'] != 1.0
            )
        else:
            transformer_common_config['embeddings_scale'] = not fairseq_args[
                'no_scale_embedding'
            ]

        # 5. Determine variant
        if fairseq_args['encoder_normalize_before']:
            transformer_common_config['variant'] = 'prelayernorm'
        elif fairseq_args['layernorm_embedding']:
            transformer_common_config['variant'] = 'bart'
        else:
            transformer_common_config['variant'] = 'aiayn'

        if self.opt['add_prefix_space']:
            transformer_common_config['bpe_add_prefix_space'] = True
        parser = ParlaiParser()
        parser.set_params(**transformer_common_config)
        opt = parser.parse_args([])

        # 6. Augment opt with additional ParlAI options
        opt['fp16'] = self.opt['fp16']
        opt['activation'] = self.opt['activation']
        opt['delimiter'] = self.opt['delimiter']
        opt['history_add_global_end_token'] = self.opt['history_add_global_end_token']
        # Makes model fp16 ready for fine-tuning, means 4 extra padding tokens.
        opt['force_fp16_tokens'] = True
        opt['converting'] = True

        return opt

    def _validate_fairseq_args(self, args: Dict[str, Any]):
        """
        Validate that fairseq args are compatible with ParlAI.
        """
        norm_key = 'encoder_normalize_before'
        assert (
            args[norm_key] == args[norm_key]
        ), "This asymmetrical transformer is not supported yet!"
        assert not (
            'layernorm_extra' in args and args['layernorm_extra']
        ), "Please handle layernorm extra"

        for k in TRANSFORMER_PARAMETER_MAPPING:
            assert (
                args[f'encoder_{k}'] == args[f'decoder_{k}']
            ), "This asymmetrical transformer is not supported yet!"

    def _load_single_fairseq_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Loads a fairseq model from file.

        :param path:
            path to file

        :return state:
            loaded fairseq state
        """
        with PathManager.open(path, "rb") as f:
            try:
                state = torch.load(
                    f, map_location=lambda s, l: default_restore_location(s, "cpu")
                )
            except ModuleNotFoundError:
                raise RuntimeError(
                    "Please install fairseq: https://github.com/pytorch/fairseq#requirements-and-installation"
                )

        return state

    def load_fairseq_checkpoint(self):
        """
        Load a checkpoint to CPU (with upgrading for backward compatibility).

        :return state:
            loaded fairseq state
        """
        paths: List[str] = self.opt['input']
        if len(paths) == 1:
            # just a single checkpoint, no fancy merges necessary
            return self._load_single_fairseq_checkpoint(paths[0])

        # many checkpoints case
        # load all the checkpoints into memory
        pieces = [self._load_single_fairseq_checkpoint(p) for p in paths]
        # store the args
        output_sd = {'args': pieces[0]['args']}

        # only need merge for the model parameters
        output_model = {}
        pieces = {k: [p['model'][k] for p in pieces] for k in pieces[0]['model'].keys()}

        for k, subpieces in pieces.items():
            if '.version' in k:
                continue
            elif '_float_tensor' in k:
                # doesn't matter for embed_positions
                output_model[k] = subpieces[0]
            elif 'out_proj.weight' in k or 'fc2.weight' in k:
                # row parallel weights
                output_model[k] = torch.cat(subpieces, dim=1)
            elif 'out_proj.bias' in k or 'fc2.bias' in k:
                # row parallel bias
                output_model[k] = subpieces[0]
            elif '_proj' in k or 'fc1' in k:
                # column parallel
                output_model[k] = torch.cat(subpieces, dim=0)
            elif '_norm' in k:
                # for norms, use any worker's copy
                output_model[k] = subpieces[0]
            elif 'embed_tokens' in k:
                output_model[k] = torch.cat(subpieces, dim=0)
            else:
                print(f"Could not handle {k}")
                __import__("ipdb").set_trace()  # FIXME
                print()

        output_sd['model'] = output_model

        return output_sd

    def convert_model_weight(self, opt: Opt) -> Dict[str, Any]:
        """
        Convert state_dict between fairseq and ParlAI.

        :param opt:
            ParlAI opt

        :return state_dict:
            return a state dict to load into ParlAI model.
        """
        # deal with embeddings
        state = self.state
        agent = self.agent
        state_dict = state['model']
        return_dict = OrderedDict()
        for each_key in state_dict.keys():
            mapped_key = each_key
            if mapped_key == 'encoder.version' or mapped_key == 'decoder.version':
                continue

            # 1. replace if embedding
            for emb in EMBEDDING_DICT_MAPPING:
                mapped_key = mapped_key.replace(emb, EMBEDDING_DICT_MAPPING[emb])

            # 2. Replace attention
            if 'encoder' in each_key and 'self_attn' in each_key:
                mapped_key = mapped_key.replace('self_attn', 'attention')
            elif 'decoder' in each_key and 'self_attn' in each_key:
                mapped_key = mapped_key.replace('self_attn', 'self_attention')
            elif 'decoder' in each_key and 'encoder_attn' in each_key:
                mapped_key = mapped_key.replace('encoder_attn', 'encoder_attention')

            # 3. Replace multihead linear layers
            #    fairseq sometimes chunks all three layers into one model weight
            if 'in_proj_weight' in mapped_key or 'in_proj_bias' in mapped_key:
                for weightorbias in {'weight', 'bias'}:
                    attention_project_name = 'in_proj_{}'.format(weightorbias)
                    if attention_project_name in mapped_key:
                        weight = state_dict[each_key]
                        size = int(weight.size(0) / 3)
                        weights = weight.split(size, 0)
                        # For Q, K, V in order
                        return_dict[
                            mapped_key.replace(
                                attention_project_name, 'q_lin.{}'.format(weightorbias)
                            )
                        ] = weights[0]
                        return_dict[
                            mapped_key.replace(
                                attention_project_name, 'k_lin.{}'.format(weightorbias)
                            )
                        ] = weights[1]
                        return_dict[
                            mapped_key.replace(
                                attention_project_name, 'v_lin.{}'.format(weightorbias)
                            )
                        ] = weights[2]
                continue
            elif (
                'v_proj' in mapped_key
                or 'k_proj' in mapped_key
                or 'q_proj' in mapped_key
            ):
                mapped_key = mapped_key.replace('v_proj', 'v_lin')
                mapped_key = mapped_key.replace('q_proj', 'q_lin')
                mapped_key = mapped_key.replace('k_proj', 'k_lin')

            # 4. Replace FFN layers
            for old, new in FFN_MAPPING.items():
                mapped_key = mapped_key.replace(old, new)

            # 5. Fix layer norms
            if 'encoder.' in mapped_key:
                mapped_key = mapped_key.replace('attention_layer_norm', 'norm1')
                mapped_key = mapped_key.replace('final_layer_norm', 'norm2')
            else:
                mapped_key = mapped_key.replace('self_attention_layer_norm', 'norm1')
                mapped_key = mapped_key.replace('encoder_attention_layer_norm', 'norm2')
                mapped_key = mapped_key.replace('final_layer_norm', 'norm3')

            for _key in ['encoder', 'decoder']:
                mapped_key = mapped_key.replace(
                    f'{_key}.layer_norm', f'{_key}.norm_embeddings'
                )
                mapped_key = mapped_key.replace(
                    f'{_key}.layernorm_embedding', f'{_key}.norm_embeddings'
                )

            weight = state_dict[each_key]
            return_dict[mapped_key] = weight

        # 6. Shuffle embedding matrix given dictionary.
        enc_emb_key = 'encoder.embeddings.weight'
        bart_dict = os.path.join(opt['datapath'], 'models/bart/bart.large/dict.txt')
        with PathManager.open(bart_dict) as f:
            offset_dict = {i: l.split()[0] for i, l in enumerate(f.readlines())}
        new_embs = return_dict[enc_emb_key].clone()
        for idx, new_idx in offset_dict.items():
            try:
                new_embs[int(new_idx) + 4] = return_dict[enc_emb_key][idx + 4]
            except ValueError:
                # if idx is not an int
                if 'madeupword' in new_idx:
                    pad_idx = int(new_idx.split('madeupword')[1])
                    new_embs[-(4 - pad_idx)] = return_dict['encoder.embeddings.weight'][
                        idx + 4
                    ]
        return_dict['encoder.embeddings.weight'] = new_embs

        # 7. Swap special tokens
        #    Fairseq swaps the bos and eos token order for seq2seq models.
        #
        #   ParlAI s2s models expect:
        #       Encoder: TOKENS </s>
        #       Decoder: <s> TOKENS <s>
        #   Fairseq models get:
        #       Encoder: TOKENS </s>
        #       Decoder: </s> TOKENS <s>
        #
        #   So we swap to get:
        #       Encoder: TOKENS </s>
        #       Decoder: </s> TOKENS <s>
        #
        size_dict = return_dict[enc_emb_key].size(0)
        if size_dict == len(agent.dict) + 1 and '<mask>' not in agent.dict:
            return_dict[enc_emb_key] = return_dict[enc_emb_key][: size_dict - 1, :]
            size_dict -= 1
        specials, words = return_dict[enc_emb_key].split([4, size_dict - 4], 0)
        bos, pad, eos, unk = specials
        if not self.opt['retain_bos_emb']:
            bos = eos
        specials = torch.stack([pad, bos, eos, unk])
        fp16_pad = (8 - (len(specials) + len(words)) % 8) % 8
        fp16_pad_ez = torch.zeros(fp16_pad, specials.size(1)).type_as(specials)
        return_dict[enc_emb_key] = torch.cat(
            [
                specials,  # special tokens
                words,  # word embeddings
                fp16_pad_ez,  # fp16 requires embeddings size to be a multiple of 8
            ],
            0,
        )

        return_dict['decoder.embeddings.weight'] = return_dict[enc_emb_key]
        return_dict['embeddings.weight'] = return_dict[enc_emb_key]

        # 8. Positional Embeddings
        if 'encoder.position_embeddings.weight' in return_dict:
            return_dict['encoder.position_embeddings.weight'] = return_dict[
                'encoder.position_embeddings.weight'
            ][2:, :]
            return_dict['decoder.position_embeddings.weight'] = return_dict[
                'decoder.position_embeddings.weight'
            ][2:, :]
        else:
            # sinusoidal embeddings
            from fairseq.modules.sinusoidal_positional_embedding import (
                SinusoidalPositionalEmbedding,
            )

            emb = SinusoidalPositionalEmbedding.get_embedding(
                128 + 2, opt['embedding_size'], 1
            )
            del return_dict['encoder.position_embeddings._float_tensor']
            del return_dict['decoder.position_embeddings._float_tensor']

            return_dict['encoder.position_embeddings.weight'] = emb[2:]
            return_dict['decoder.position_embeddings.weight'] = emb[2:]

        return_dict['START'] = torch.LongTensor([1])  # type: ignore
        return return_dict

    def print_agent_act(self):
        """
        Print a sample act from the converted agent.
        """
        self.agent.observe(
            {'text': "What's your favorite kind of ramen?", 'episode_done': False}
        )
        print(self.agent.act())


if __name__ == '__main__':
    ConversionScript.main()
