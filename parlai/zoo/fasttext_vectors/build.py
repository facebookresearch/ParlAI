# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Fastext vectors, e.g. use with filename "models:fasttext_vectors/wiki.en.vec"
"""

import torchtext.vocab as vocab


def download(datapath):
    vocab.FastText(language='en', cache=datapath + '/models/fasttext_vectors')
