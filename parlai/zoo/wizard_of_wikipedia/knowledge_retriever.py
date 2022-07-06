#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained retrieval model on the Wizard of Wikipedia dialogue task.
"""
from parlai.core.build_data import download_models

from .end2end_generator import download as generator_download
from .full_dialogue_retrieval_model import download as retrieval_download


def download(datapath):
    opt = {'datapath': datapath}

    # download all relevant wizard of wikipedia models
    generator_download(datapath)
    retrieval_download(datapath)

    # now download knowledge retriever
    fnames = ['knowledge_retriever.tgz']
    opt['model_type'] = 'knowledge_retriever'
    download_models(opt, fnames, 'wizard_of_wikipedia', version='v3.0')
