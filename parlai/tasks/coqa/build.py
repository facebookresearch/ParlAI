#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import json


def make_parlai_format(outpath, dtype, data):
    print('building parlai:' + dtype)
    fout = open(os.path.join(outpath, dtype + '.txt'), 'w')
    for each in data:
        s = ""
        for question, ans in zip(each['questions'], each['answers']):
            s += "text:"
            if question['turn_id'] == 1:
                s += each['story'] + '\n'
            s += question['input_text'] + '\t\t'
            s += "labels:" + ans['input_text']
            if question['turn_id'] < len(each['questions']):
                s += '\n'
            s += '\t\t' + "episode_done:True" + '\n'
        fout.write(s)
    fout.close()


def build(opt):
    dpath = os.path.join(opt['datapath'], 'CoQA')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        tfname = 'coqa-train-v1.0.json'
        dfname = 'coqa-dev-v1.0.json'
        url = 'https://nlp.stanford.edu/data/coqa/'
        build_data.download(url + tfname, dpath, tfname)
        build_data.download(url + dfname, dpath, dfname)
        with open(os.path.join(dpath, tfname)) as f:
            data = json.load(f)['data']
        train_p = 0.8
        valid_p = 0.2
        assert train_p > 0
        assert valid_p > 0

        data_len = len(data)
        first_valid = int(data_len * train_p)
        make_parlai_format(dpath, 'train', data[:first_valid])
        make_parlai_format(dpath, 'valid', data[first_valid:])
        with open(os.path.join(dpath, dfname)) as f:
            data = json.load(f)['data']
        make_parlai_format(dpath, 'test', data)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
