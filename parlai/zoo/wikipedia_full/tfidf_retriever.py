#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
TFIDF retriever for Wikipedia (for use with e.g. DrQA+SQuAD).
"""

from parlai.core.build_data import download_models


def download(datapath):
    opt = {'datapath': datapath}
    fnames = ['model.tgz']
    opt['model_type'] = 'tfidf_retriever'  # for builder
    download_models(opt, fnames, 'wikipedia_full', use_model_type=True)
