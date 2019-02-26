#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This downloads a pretrained language model BERT
(Devlin et al. https://arxiv.org/abs/1810.04805). It relies on the pytorch
implementation provided by Hugging Face
(https://github.com/huggingface/pytorch-pretrained-BERT).
"""

import parlai.core.build_data as build_data
import os


def download(datapath, version='v1.0'):
    dpath = os.path.join(datapath, 'models', 'bert_models')

    if not build_data.built(dpath, version):
        print('[downloading BERT models: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fnames = ['bert-base-uncased.tar.gz', 'bert-base-uncased-vocab.txt']
        for fname in fnames:
            url = ('https://s3.amazonaws.com/models.huggingface.co/bert/' +
                   fname)
            build_data.download(url, dpath, fname)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
