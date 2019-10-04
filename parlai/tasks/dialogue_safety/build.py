#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import parlai.core.build_data as build_data


SINGLE_TURN_DATA = 'single_turn_safety.json'
MULTI_TURN_DATA = 'multi_turn_safety.json'


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
        fnames = [SINGLE_TURN_DATA, MULTI_TURN_DATA]
        for fname in fnames:
            url = 'http://parl.ai/downloads/dialogue_safety/' + fname
            build_data.download(url, dpath, fname)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
