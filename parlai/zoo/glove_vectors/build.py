# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""DrQA model (reader only) for SQuAD.
"""

from parlai.core.build_data import download_models
import copy

def download(datapath):
    opt  = { 'datapath': datapath }
    fnames = ['glove.840B.300d.zip']
    download_models(opt, fnames, 'glove_vectors', use_model_type=False,
                    path = "http://nlp.stanford.edu/data")
            
