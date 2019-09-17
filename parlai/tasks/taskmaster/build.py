#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import parlai.core.build_data as build_data
import os


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'taskmaster-1')
    # define version if any
    version = "1.0"

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the data.
        gsl_url = 'https://storage.googleapis.com/dialog-data-corpus/TASKMASTER-1-2019/'
        fname_self_dialogs = 'self-dialogs.json'
        fname_woz_dialogs = 'woz-dialogs.json'
        url_self_dialogs = gsl_url + fname_self_dialogs  # dataset URL
        url_woz_dialogs = gsl_url + fname_woz_dialogs  # dataset URL
        build_data.download(url_self_dialogs, dpath, fname_self_dialogs)
        build_data.download(url_woz_dialogs, dpath, fname_woz_dialogs)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
