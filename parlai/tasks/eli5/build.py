#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import subprocess
from os.path import join as pjoin
from os.path import isfile, isdir


# pre-computed files
RESOURCES = [
    # wet.paths.gz is false because the archive format is not recognized
    # It gets unzipped with subprocess after RESOURCES are downloaded.
    DownloadableFile(
        'https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2018-34/wet.paths.gz',
        'wet.paths.gz',
        'e3a8addc6a33b54b1dd6488a98c875851ef1aca3b80133d39f6897330a8835fb',
        zipped=False,
    ),
    DownloadableFile(
        'https://dl.fbaipublicfiles.com/eli5qa/explainlikeimfive_ccrawl_ids.json.gz',
        'explainlikeimfive_ccrawl_ids.json.gz',
        '59cd7b6a8580421aecae66cd33d065073f2abf21d86097b3262bd460a7a14f0d',
        zipped=False,
    ),
    DownloadableFile(
        'https://dl.fbaipublicfiles.com/eli5qa/explainlikeimfive_unigram_counts.json',
        'explainlikeimfive_unigram_counts.json',
        '0433a4dda7532ba1dae2f5b6bf70cd5ab91fd2772f75e99b4c15c2e04ba80dfd',
        zipped=False,
    ),
]


# Setup a directory called dir_name at dpath
def setup_dir(dpath, dir_name):
    if not isdir(pjoin(dpath, dir_name)):
        subprocess.run(['mkdir', pjoin(dpath, dir_name)])


def build(opt):
    dpath = pjoin(opt['datapath'], 'eli5')
    pre_computed_path = pjoin(dpath, 'pre_computed')
    processed_path = pjoin(dpath, 'processed_data')

    version = '1.0'

    if not build_data.built(dpath, version_string=version):
        # Setup directory folders
        setup_dir(dpath, 'tmp')
        setup_dir(dpath, 'pre_computed')
        setup_dir(dpath, 'processed_data')
        setup_dir(processed_path, 'collected_docs')

        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the pre-computed data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(pre_computed_path)
            if downloadable_file.file_name.endswith('.gz'):
                # Check file hasn't already been unzipped
                if not isfile(
                    pjoin(pre_computed_path, downloadable_file.file_name[:-3])
                ):
                    print(f'Unzipping {downloadable_file.file_name}')
                    subprocess.run(
                        [
                            'gunzip',
                            pjoin(pre_computed_path, downloadable_file.file_name),
                        ],
                        stdout=subprocess.PIPE,
                    )

    # Check that wet.paths and eli5 ccrawl ids have been unzipped
    paths_unzipped = True
    eli_ids_unzipped = True
    if not isfile(pjoin(pre_computed_path, 'wet.paths')):
        print(
            f'Unzip wet.paths.gz located at {pre_computed_path} with command:\n',
            'gunzip wet.paths.gz',
        )
        paths_unzipped = False
    if not isfile(pjoin(pre_computed_path, 'explainlikeimfive_ccrawl_ids.json')):
        print(
            f'Unzip explainlikeimfive_ccrawl_ids.json.gz located at {pre_computed_path} with command:\n',
            'gunzip explainlikeimfive_ccrawl_ids.json.gz',
        )
        eli_ids_unzipped = False
    if paths_unzipped and eli_ids_unzipped:
        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)

    # Check that reddit file is there for eli5
    sel_path = pjoin(processed_path, 'selected_15_1')
    test_train_valid_check = (
        isfile(pjoin(sel_path, 'explainlikeimfive_test.json'))
        and isfile(pjoin(sel_path, 'explainlikeimfive_train.json'))
        and isfile(pjoin(sel_path, 'explainlikeimfive_valid.json'))
    )
    if not test_train_valid_check:
        print(
            '\nDid not find one of the ELI5 test, train, valid json files.',
            '\nPlease make sure you have run the steps in',
            'https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/eli5\n',
        )
