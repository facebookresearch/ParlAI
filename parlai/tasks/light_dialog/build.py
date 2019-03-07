#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.tasks.light_dialog.builder import build_from_db


def build(opt):
    version = 'v2.01'
    # download pickled database
    dpath = os.path.join(opt['datapath'], 'light_dialogue')
    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        # Download the data.
        url = ('https://dl.fbaipublicfiles.com/parlai/light/' +
               'light-dialog-processed-small7.pkl')
        fname = 'light_data.pkl'
        build_data.download(url, dpath, fname)
        # Download the unseen data.
        url = 'https://dl.fbaipublicfiles.com/parlai/light/light-unseen-processed2.pkl'
        fname = 'light_unseen_data.pkl'
        build_data.download(url, dpath, fname)
        # Mark the data as built.
        build_data.mark_done(dpath, version)

    # create particular instance of dataset depending on flags..
    fields = [
        'setting',
        'objects',
        'person_names',
        'persona',
        'emote',
        'speech',
        'action',
        'repeat',
        'cands',
        'current_self_output',
        'clip_cands',
    ]
    fpath = ''
    for f in fields:
        fpath += f + str(opt['light_use_' + f]) + "_"
    dpath2 = os.path.join(opt['datapath'], 'light_dialogue', fpath[:-1])
    if not build_data.built(dpath2, version):
        if build_data.built(dpath2):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath2)
        build_data.make_dir(dpath2)
        fname = 'light_data.pkl'
        fname2 = 'light_unseen_data.pkl'
        build_from_db(opt, dpath, dpath2, fname, fname2)
        # Mark the data as built.
        build_data.mark_done(dpath2, version)
