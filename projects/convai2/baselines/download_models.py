# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os

def download(opt, path, fname, version='1.0'):
    fshort = fname[:fname.find('.')] if '.' in fname else fname
    dpath = os.path.join(opt['datapath'], 'models', path, fshort)

    if not build_data.built(dpath, version):
        print('[downloading: ' + dpath + '/' + fname + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        url = 'https://s3.amazonaws.com/fair-data/parlai/_models/convai2/' + fname
        build_data.download(url, dpath, fname)
        if '.tgz' in fname or '.gz' in fname:
            build_data.untar(dpath, fname)
        # Mark the data as built.
        build_data.mark_done(dpath, version)
