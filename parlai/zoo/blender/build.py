#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Blender models: please see <parl.ai/project/blender>.
"""

import os
from parlai.core.build_data import download_models, built


def print_blender():
    curr_dir = os.path.dirname(__file__)
    txt_file = os.path.join(curr_dir, 'blender_ascii.txt')
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            print(line[:-1])


def build(datapath, fname, model_type, version):
    opt = {'datapath': datapath}
    opt['model_type'] = model_type
    dpath = os.path.join(datapath, 'models', 'blender', model_type)
    if not built(dpath, version):
        print_blender()
    download_models(opt, [fname], 'blender', version=version, use_model_type=False)
