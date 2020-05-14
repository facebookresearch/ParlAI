#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os


def build(opt):
    version = 'v6.0'
    dpath = os.path.join(opt['datapath'], 'rephrase_sentences')

    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fnames = [
            'rephrase_sentences_train_0703.txt',
            'rephrase_sentences_valid_0703.txt',
            'rephrase_sentences_test_0703.txt',
            'choose_sentence_train.txt',
            'choose_sentence_valid.txt',
            'choose_sentence_test.txt',
        ]
        for fname in fnames:
            url = 'http://parl.ai/downloads/projects/rephrase_sentences/' + fname
            build_data.download(url, dpath, fname)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
