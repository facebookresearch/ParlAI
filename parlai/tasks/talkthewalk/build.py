#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import json
from .ttw.dict import Dictionary


def build(opt):
    dpath = os.path.join(opt['datapath'], 'TalkTheWalk')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = 'talkthewalk.tgz'
        url = 'https://s3.amazonaws.com/fair-data/parlai/projects/talkthewalk/' + fname
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)


        train_set = json.load(open(os.path.join(dpath, 'talkthewalk.train.json')))
        valid_set = json.load(open(os.path.join(dpath, 'talkthewalk.valid.json')))
        test_set = json.load(open(os.path.join(dpath, 'talkthewalk.test.json')))

        dictionary = Dictionary()
        for set in [train_set, valid_set, test_set]:
            for config in set:
                for msg in config['dialog']:
                    if msg['id'] == 'Tourist':
                        if msg['text'] not in ['ACTION:TURNLEFT', 'ACTION:TURNRIGHT', 'ACTION:FORWARD']:
                            if len(msg['text'].split(' ')) > 2:
                                dictionary.add(msg['text'])

        dictionary.save(os.path.join(dpath, 'dict.txt'))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
