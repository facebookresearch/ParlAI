#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fastext vectors, e.g. use with filename "models:fasttext_vectors/wiki.en.vec"
"""

import torchtext.vocab as vocab


def download(datapath):
    vocab.FastText(language='en', cache=datapath + '/models/fasttext_vectors')
