#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fastext vectors, e.g. use with filename "models:fasttext_vectors/wiki.en.vec"
"""

import torchtext.vocab as vocab
from parlai.core.build_data import modelzoo_path

URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec'


def download(datapath):
    return vocab.Vectors(
        name='wiki.en.vec',
        url=URL,
        cache=modelzoo_path(datapath, 'models:fasttext_vectors'),
    )
