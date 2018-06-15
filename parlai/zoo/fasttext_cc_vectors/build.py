# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Fastext Common Crawl vectors, e.g. use with filename
"models:fasttext_cc_vectors/crawl-300d-2M.vec"
"""

import torchtext.vocab as vocab


def download(datapath):
    embs = vocab.Vectors(
        name='crawl-300d-2M.vec',
        url='https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip',
        cache=datapath + '/models/fasttext_cc_vectors'
    )
