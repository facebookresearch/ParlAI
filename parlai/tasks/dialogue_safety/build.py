#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import parlai.core.build_data as build_data

URLS = ['http://parl.ai/downloads/dialogue_safety/single_turn_safety.json',
        'http://parl.ai/downloads/dialogue_safety/multi_turn_safety.json']
FILE_NAMES = ['single_turn_safety.json',
                'multi_turn_safety.json']
SHA256 = ['f3a46265aa639cfa4b55d2be4dca4be1c596acb5e8f94d7e0041e1a54cedd4cd',
        'e3e577f456d63d51eb7b5f98ffd251ad695476f186d422fa8de1a177742fa7b6']

def build(datapath):
    version = 'v1.0'
    dpath = os.path.join(datapath, 'dialogue_safety')

    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
