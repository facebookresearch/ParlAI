#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/wikipedia/wiki_full_extracted.tgz',
        'wiki_full_extracted.tgz',
        'c8f5ed2a8a81e50bbcef5b7a8b9728c960254ef0d4cfc00e47211a3ce6b0e1fb',
    ),
    DownloadableFile(
        'http://parl.ai/downloads/wikipedia/summaries.tgz',
        'summaries.tgz',
        'e8e1c35d33e28a1b85e52adf1fe106938543d384578c841029384a2d6ec2b259',
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'wikipedia')
    task = opt.get('task', 'wikipedia:all')
    extract_full = task.split(':')[-1] == 'all'
    if extract_full:
        dpath = os.path.join(dpath, 'full')
    else:
        dpath = os.path.join(dpath, 'summary')
    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')
        build_data.make_dir(dpath)
        if extract_full:
            RESOURCES[0].download_file(dpath)
        else:
            RESOURCES[1].download_file(dpath)
        build_data.mark_done(dpath)
