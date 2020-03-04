#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.tasks.light_dialog.builder import build_from_db
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/light/light-dialog-processed-small7.pkl',
        'light_data.pkl',
        '7c83cf49818586db9999ea67a4a6ad087afbd91c26ed629a9f00e21d0b84058f',
        zipped=False,
    ),
    DownloadableFile(
        'http://parl.ai/downloads/light/light-unseen-processed2.pkl',
        'light_unseen_data.pkl',
        '489b98d08dd94eaf1ba95439d04200ccc54623ade056839f87a5c4207bc5699c',
        zipped=False,
    ),
    DownloadableFile(
        'http://parl.ai/downloads/light/light-environment.pkl',
        'light_environment.pkl',
        '162389202f22063e1c32af7f9261aac13d20fc05598388d1e9748735996ec016',
        zipped=False,
    ),
]


def download(opt):
    version = 'v2.03'
    # download pickled database
    dpath = os.path.join(opt['datapath'], 'light_dialogue')
    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version)

    return dpath, version


def build(opt):
    dpath, version = download(opt)
    if 'light_use_speech_prefix' not in opt:
        opt['light_use_speech_prefix'] = True
    # create particular instance of dataset depending on flags..
    fields = [
        'taskname',
        'setting',
        'objects',
        'person_names',
        'persona',
        'emote',
        'speech',
        'action',
        'affordances',
        'repeat',
        'cands',
        'current_self_output',
        'clip_cands',
        'speech_prefix',
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
