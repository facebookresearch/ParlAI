# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import os

from parlai.core import build_data


def build(opt):
    dpath = os.path.join(opt['datapath'], 'dialog-bAbI-plus')
    fname = "dialog-bAbI-plus.zip"
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        url = "https://drive.google.com/uc?export=download&id=0B2MvoQfXtqZmMTJqclpBdGN2bmc"
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)

        build_data.mark_done(dpath, version)