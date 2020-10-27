#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
from parlai.core.build_data import DownloadableFile
from .utils.trade_proc import trade_process
from .utils.reformat import reformat_dial
import os

RESOURCES = [
    DownloadableFile(
        'https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y',
        'MULTIWOZ2.1.zip',
        'd377a176f5ec82dc9f6a97e4653d4eddc6cad917704c1aaaa5a8ee3e79f63a8e',
    )
]


def build(opt):
    download_path = os.path.join(opt['datapath'], 'multiwoz21_trade')
    data_dir = os.path.join(download_path, 'MULTIWOZ2.1')
    version = '1.0'

    if not build_data.built(download_path, version_string=version):
        print('[building data: ' + download_path + ']')

        # make a clean directory if needed
        if build_data.built(download_path):

            # an older version exists, so remove these outdated files.
            build_data.remove_dir(download_path)
        build_data.make_dir(download_path)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(download_path)

        # process the data with TRADE's code, if it does not exist
        if not os.path.exists(os.path.join(data_dir, 'data_trade.json')):
            trade_process(data_dir)

        # reformat data for both DST and response generation
        if not os.path.exists(os.path.join(data_dir,
'data_trade_reformat.json')):
            reformat_dial(data_dir)

        # mark the data as built
        build_data.mark_done(download_path, version_string=version)

