# Copyright (c) 2017-present, Moscow Institute of Physics and Technology.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import parlai.core.build_data as build_data
import os


def build(opt):
    import json
    data_path = os.path.join(opt['datapath'], 'ConvAIChitChat')
    version = '1501534800'

    if not build_data.built(data_path, version_string=version):
        print('[building data: ' + data_path + ']')

        if build_data.built(data_path):
            build_data.remove_dir(data_path)
        build_data.make_dir(data_path)

        fname = 'data_' + version + '.tar.gz'
        url = 'https://raw.githubusercontent.com/deepmipt/turing-data/master/' + fname
        build_data.download(url, data_path, fname)
        build_data.untar(data_path, fname)

        build_data.mark_done(data_path, version_string=version)
