# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import parlai.core.build_data as build_data
import os

def build(opt):
    dpath = os.path.join(opt['datapath'], 'negotiation')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data from github
        fname = 'negotiation.zip'
        url = ('https://github.com/facebookresearch/end-to-end-negotiator/'
               'archive/master.zip')
        print('[downloading data from: ' + url + ']')
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)

        # Mark as done
        build_data.mark_done(dpath, version_string=version)
