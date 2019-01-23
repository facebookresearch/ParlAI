#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'wikipedia')
    task = opt.get('task', 'wikipedia:all')
    extract_full = task.split(':')[-1] == 'all'
    if extract_full:
        dpath = os.path.join(dpath, 'full')
        fname = 'wiki_full_extracted.tgz'
    else:
        dpath = os.path.join(dpath, 'summary')
        fname = "summaries.tgz"
    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')
        build_data.make_dir(dpath)
        url = 'http://parl.ai/downloads/wikipedia/' + fname
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)
        build_data.mark_done(dpath)
