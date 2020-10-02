#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

ROOT_URL = 'https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset/tree/master/data/paper_splits/'

SENTENCE_ANNOTATIONS_PATH = 'splits_annotated_at_sentence_level'
SENTENCE_ANNOTATIONS_PATH = 'splits_annotated_at_turn_level'

RESOURCES = {
    # raw data files
    DownloadableFile(
        f'{ROOT_URL}/ontology/sports.json',
        'sports.onto.json',
        '52f9bbb86ebd9e2b3916185ad4e3e9b8b77d2164d96bd3b98ad67cbaa653757d',
        zipped=False,
    )
}


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'taskmaster-2')
    # define version if any
    version = "1.1"

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
