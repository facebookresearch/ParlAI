#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'wizard_of_wikipedia')
    fname = 'wizard_of_wikipedia.tgz'
    version = '1.0'
    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        url = 'http://parl.ai/downloads/wizard_of_wikipedia/' + fname
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)
        build_data.mark_done(dpath, version)
