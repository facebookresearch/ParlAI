#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import json


def build(opt):
    version = 'v0.9'
    dpath = os.path.join(opt['datapath'], 'VisDial-v0.9')

    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')

        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname1 = 'visdial_0.9_train.zip'
        fname2 = 'visdial_0.9_val.zip'

        url = 'https://computing.ece.vt.edu/~abhshkdz/data/visdial/'
        build_data.download(url + fname1, dpath, fname1)
        build_data.download(url + fname2, dpath, fname2)

        build_data.untar(dpath, fname1)
        build_data.untar(dpath, fname2)

        print('processing unpacked files')
        # Use 1000 examples from training set as validation.
        json1 = os.path.join(dpath, fname1.rsplit('.', 1)[0] + '.json')
        with open(json1) as t_json:
            train_data = json.load(t_json)

        valid_data = train_data.copy()
        valid_data['data'] = train_data['data'].copy()
        valid_data['data']['dialogs'] = []

        # Use constant stride to pick examples.
        num_valid = 1000
        total = len(train_data['data']['dialogs'])
        step = total // (num_valid - 1)
        for i in range(total - 1, 0, -step)[:num_valid]:
            valid_data['data']['dialogs'].append(train_data['data']['dialogs'][i])
            del train_data['data']['dialogs'][i]

        train_json = json1.rsplit('.', 1)[0] + '_train.json'
        valid_json = json1.rsplit('.', 1)[0] + '_valid.json'
        with open(train_json, 'w') as t_out, open(valid_json, 'w') as v_out:
            json.dump(train_data, t_out)
            json.dump(valid_data, v_out)
        os.remove(json1)

        # Use validation data as test.
        json2 = os.path.join(dpath, fname2.rsplit('.', 1)[0] + '.json')
        test_json = json2.rsplit('.', 1)[0] + '_test.json'
        build_data.move(json2, test_json)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
