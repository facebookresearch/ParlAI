#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import subprocess
import os

from os.path import join as pjoin
from os.path import isfile
from os.path import isdir


# pre-computed files
RESOURCES = [
    DownloadableFile(
        'https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2018-34/wet.paths.gz',
        'wet.paths.gz',
        'e3a8addc6a33b54b1dd6488a98c875851ef1aca3b80133d39f6897330a8835fb',
    ),
    DownloadableFile(
        'https://dl.fbaipublicfiles.com/eli5qa/explainlikeimfive_ccrawl_ids.json.gz',
        'explainlikeimfive_ccrawl_ids.json.gz',
        '7fafae2c33fafc80d65bd55c8026c9f6a9e50a7eb4f98a7f0ff2780bf1459444',
    ),
    DownloadableFile(
        'https://dl.fbaipublicfiles.com/eli5qa/explainlikeimfive_unigram_counts.json',
        'explainlikeimfive_unigram_counts.json',
        '0433a4dda7532ba1dae2f5b6bf70cd5ab91fd2772f75e99b4c15c2e04ba80dfd',
    ),

]

# Setup a directory called dir_name at dpath
def setup_dir(dpath, dir_name):
    if not isdir(pjoin(dpath, 'tmp')):
            subprocess.run(['mkdir', pjoin(dpath, 'tmp')])

def build(opt):
    dpath = pjoin(opt['datapath'], 'eli5')
    pre_computed_path = pjoin(dpath, 'pre_computed')
    processed_path = pjoin(dpath, 'processed_data')
    docs_path = pjoin(processed_path, 'collected_data')

    version = '1.0'

    

    if not build_data.built(dpath, version_string=version):
        print('[building image data: ' + dpath + ']')

        # Setup directory folders
        setup_dir(dpath, 'tmp')
        setup_dir(dpath, 'eli5')
        setup_dir(dpath, 'processed_data')
        setup_dir(processed_path, 'collected_data')
        

        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the pre-computed data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(pre_computed_path)
            if downloadable_file.file_name.endswith('.gz'):
                subprocess.run(['gunzip', pjoin(pre_computed_path, downloadable_file.file_name)])

        # Check that wet.paths and eli5 ccrawl ids have been unzipped
        if not isfile(pjoin(pre_computed_path, 'wet.paths')):
            print(f'Unzip wet.paths.gz located at {pre_computed_path} with command:\n',
                 'gunzip wet.paths.gz')
        if not isfile(pjoin(pre_computed_path, 'explainlikeimfive_ccrawl_ids.json')):
            print(f'Unzip explainlikeimfive_ccrawl_ids.json.gz located at {pre_computed_path} with command:\n',
                 'gunzip explainlikeimfive_ccrawl_ids.json.gz')


        # Check that reddit file is there for eli5
        sel_path = pjoin(processed_path, 'selected_15_1')
        test_train_valid_check = isfile(pjoin(sel_path, 'explainlikeimfive_test.json')) \
                                 and isfile(pjoin(sel_path, 'explainlikeimfive_train.json')) \
                                 and isfile(pjoin(sel_path, 'explainlikeimfive_valid.json'))
        if not test_train_valid_check:
            print('Did not find one of the ELI5 test, train, valid json files. \n',
            'Please make sure you have run the steps in https://github.com/facebookresearch/ELI5')
        
        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
