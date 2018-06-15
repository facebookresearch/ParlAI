# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""DrQA model (reader only) for SQuAD.
"""

from parlai.core.build_data import download_models
import torchtext.vocab as vocab

def download(datapath):
    embs = vocab.FastText(language='en',
                          cache=datapath + '/models/fasttext_vectors')            
