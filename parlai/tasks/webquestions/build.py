#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import json
import os
import re

STRIP_CHARS = ' ()"()'


def parse_ans(a):
    a = a.lstrip('(list')
    ans = ''
    for a in re.split(r'\(description', a):
        a = a.strip(STRIP_CHARS)
        ans = ans + '|' + a
    return ans.lstrip('|')


def create_fb_format(outpath, dtype, inpath):
    print('building fbformat:' + dtype)
    with open(inpath) as data_file:
        data = json.load(data_file)
    fout = open(os.path.join(outpath, dtype + '.txt'), 'w')
    for i in range(len(data)):
        q = data[i]['utterance']
        a = parse_ans(data[i]['targetValue'])
        s = '1 ' + q + '\t' + a
        fout.write(s + '\n')
    fout.close()


def build(opt):
    dpath = os.path.join(opt['datapath'], 'WebQuestions')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        url = ('https://worksheets.codalab.org/rest/bundles/' +
               '0x4a763f8cde224c2da592b75f29e2f5c2/contents/blob/')
        build_data.download(url, dpath, 'train.json')

        url = ('https://worksheets.codalab.org/rest/bundles/' +
               '0xe7bac352fce7448c9ef238fb0a297ec2/contents/blob/')
        build_data.download(url, dpath, 'test.json')

        create_fb_format(dpath, 'train', os.path.join(dpath, 'train.json'))
        create_fb_format(dpath, 'valid', os.path.join(dpath, 'train.json'))
        create_fb_format(dpath, 'test', os.path.join(dpath, 'test.json'))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
