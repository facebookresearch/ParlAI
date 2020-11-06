#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained Transformer-based classifier for classification of sensitive topics.

Example command: ``` parlai eval_model -mf zoo:sensitive_topics_classifier/model -t
sensitive_topics_evaluation -dt valid -bs 16 ```
"""
from parlai.core.build_data import built, download_models, get_model_dir
import os
import os.path


def download(datapath):
    model_name = 'sensitive_topics_classifier'
    mdir = os.path.join(get_model_dir(datapath), model_name)
    version = 'v1'
    if not built(mdir, version):
        opt = {'datapath': datapath}
        fnames = ['sensitive_topics_classifier2.tgz']
        download_models(
            opt,
            fnames,
            model_name,
            version=version,
            use_model_type=False,
            flatten_tar=True,
        )
