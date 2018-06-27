# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.params import ParlaiParser
from parlai.scripts.build_dict import setup_args, build_dict as main_build_dict

import os


def download(datapath):
    DICT_PATH = os.path.join(datapath, 'models', 'twitter', 'dict', 'dict_30k')
    # don't actually download--build it
    parser = setup_args(ParlaiParser())
    # first build on standard train and validation
    parser.set_defaults(
        task='twitter',
        dict_lower=True,
        dict_file=DICT_PATH,
        dict_maxtokens=30000,
    )
    opt = parser.parse_args(args='')
    return main_build_dict(opt)
