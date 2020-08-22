#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import parlai.core.params as params
import parlai.core.build_data as build_data


URL_ROOT = 'https://parl.ai/downloads/controllable_dialogue/'
FOLDER_NAME = 'controllable_dialogue'


def build(opt):
    dpath = os.path.join(opt['datapath'], FOLDER_NAME)
    # version 1.0: initial release
    # version 1.1: add evaluation logs
    # version 1.2: add reproducible evaluation logs
    version = '1.2'

    if not build_data.built(dpath, version_string=version):
        if build_data.built(dpath):
            # older version exists, so remove the outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # first download the data files
        fname_data = 'data_v1.tar.gz'
        build_data.download(URL_ROOT + fname_data, dpath, fname_data)
        build_data.untar(dpath, fname_data)

        # next download the wordstats files
        fname_wordstats = 'wordstats_v1.tar.gz'
        build_data.download(URL_ROOT + fname_wordstats, dpath, fname_wordstats)
        build_data.untar(dpath, fname_wordstats)

        # next download the evaluation logs
        fname_evallogs = 'evaluationlogs_v1.tar.gz'
        build_data.download(URL_ROOT + fname_evallogs, dpath, fname_evallogs)
        build_data.untar(dpath, fname_evallogs)

        # and the reproducible logs.
        # for more info see https://github.com/facebookresearch/ParlAI/issues/2855
        fname_evallogs = 'evaluation_logs_reproducible_v1.tar.gz'
        build_data.download(URL_ROOT + fname_evallogs, dpath, fname_evallogs)
        build_data.untar(dpath, fname_evallogs)

        print("Data has been placed in " + dpath)

        build_data.mark_done(dpath, version)


def make_path(opt, fname):
    return os.path.join(opt['datapath'], FOLDER_NAME, fname)


if __name__ == '__main__':
    opt = params.ParlaiParser().parse_args()
    build(opt)
