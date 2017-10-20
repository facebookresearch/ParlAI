# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import random

random.seed(1)


def _train_test_valid_split(dialogs):
    import math
    random.shuffle(dialogs)
    count = len(dialogs)
    border_valid = math.floor(count * 0.1)
    valid = dialogs[:border_valid]
    border_test = border_valid + math.floor(count * 0.1)
    test = dialogs[border_valid:border_test]
    train = dialogs[border_test:]
    return train, test, valid


def build(opt):
    import json
    dpath = os.path.join(opt['datapath'], 'ConvAIChitChat')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        fname = 'train_full.json'
        url = 'https://raw.githubusercontent.com/deepmipt/turing-data/master/' + fname
        build_data.download(url, dpath, fname)

        with open(os.path.join(dpath, fname)) as dataset:
            dialogs = json.load(dataset)
            train, test, valid = _train_test_valid_split(dialogs)
            with open(os.path.join(dpath, "valid.json"), 'w') as outfile:
                json.dump(valid, outfile)
            with open(os.path.join(dpath, "test.json"), 'w') as outfile:
                json.dump(test, outfile)
            with open(os.path.join(dpath, "train.json"), 'w') as outfile:
                json.dump(train, outfile)

        build_data.mark_done(dpath, version_string=version)