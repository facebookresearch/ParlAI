#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fastext Common Crawl vectors, e.g. use with filename
"models:fasttext_cc_vectors/crawl-300d-2M.vec"
"""

import torchtext.vocab as vocab

url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip'


def download(datapath):
    vocab.Vectors(
        name='crawl-300d-2M.vec',
        url=url,
        cache=datapath + '/models/fasttext_cc_vectors'
    )
