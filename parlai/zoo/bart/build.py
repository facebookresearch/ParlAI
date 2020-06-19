#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This downloads a pretrained language model BART (Lewis et al.

https://arxiv.org/abs/1910.13461). It requires you to run
a conversion script to map model weights (conversion script not provided publicly).
"""

import parlai.core.build_data as build_data
from parlai.scripts.convert_fairseq_to_parlai import ConversionScript
import os

CONVERSION_ARGS = {
    'add_prefix_space': False,
    'activation': 'gelu',
    'tokenizer': 'gpt2',
    'delimiter': '\n',
    'retain_bos_emb': True,
    'model': 'bart',
    'fp16': True,
    'history_add_global_end_token': 'end'
}


def download(datapath, version='v1.0'):
    dpath = os.path.join(datapath, 'models', 'bart_models')

    if not build_data.built(dpath, version):
        print('[downloading BART models: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fnames = ['bart.large.tar.gz', 'bart.large.mnli.tar.gz', 'bart.large.cnn.tar.gz']
        for fname in fnames:
            url = f'http://dl.fbaipublicfiles.com/fairseq/models/{fname}'
            build_data.download(url, dpath, fname)
            build_data.untar(dpath, fname)
            args = CONVERSION_ARGS.copy()
            args['input'] = [os.path.join(dpath, fname.replace('.tar.gz', ''), 'model.pt')]
            args['output'] = os.path.join(dpath, f"{fname.replace('.tar.gz', '')}", 'model')
            ConversionScript.main(**args)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
