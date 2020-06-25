#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This downloads a pretrained language model BART (Lewis et al.

https://arxiv.org/abs/1910.13461). It requires you to run a conversion script to map
model weights (conversion script not provided publicly).
"""

from parlai.agents.bart.convert_fairseq_to_parlai import ConversionScript
import parlai.core.build_data as build_data
import os

CONVERSION_ARGS = {
    'add_prefix_space': False,
    'activation': 'gelu',
    'tokenizer': 'gpt2',
    'delimiter': '\n',
    'retain_bos_emb': True,
    'model': 'bart',
    'fp16': True,
    'history_add_global_end_token': None,
}

BART_ARGS = {
    'embedding_size': 1024,
    'ffn_size': 4096,
    'dropout': 0.1,
    'attention_dropout': 0.1,
    'n_heads': 16,
    'n_positions': 1024,
    'variant': 'bart',
    'activation': 'gelu',
    'n_encoder_layers': 12,
    'n_decoder_layers': 12,
    'force_fp16_tokens': True,
    'fp16': True,
    'dict_tokenizer': 'gpt2',
    'embeddings_scale': False,
    'history_add_global_end_token': None,
    'learn_positional_embeddings': True,
}


def download(datapath, version='v1.0'):
    dpath = os.path.join(datapath, 'models', 'bart')

    if not build_data.built(dpath, version):
        print('[downloading BART models: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        model_name = 'bart.large'
        url = f'http://dl.fbaipublicfiles.com/fairseq/models/{model_name}.tar.gz'
        build_data.download(url, dpath, f'{model_name}.tar.gz')
        build_data.untar(dpath, f'{model_name}.tar.gz')
        args = CONVERSION_ARGS.copy()
        args['input'] = [os.path.join(dpath, model_name, 'model.pt')]
        args['output'] = os.path.join(dpath, model_name.replace('.', '_'), 'model')
        ConversionScript.main(**args)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
