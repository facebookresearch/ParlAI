#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from parlai.core.build_data import download_models


def download(datapath):
    opt = {'datapath': datapath}
    fnames = ['hh131k_hb60k_fb60k_st1k_v1.tar.gz']
    download_models(opt, fnames, 'self_feeding', version='v1.0', use_model_type=False)
