#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def create_fb_format(outpath, dtype, inpath, inpath2):
    print('building fbformat:' + dtype)
    fout = open(os.path.join(outpath, dtype + '.txt'), 'w')
    with open(inpath + '.tsv') as f:
        lines = [line.strip('\n') for line in f]
    if inpath2 is None:
        fname_ans = inpath + '.ans'
    else:
        fname_ans = inpath2
    with open(fname_ans) as f:
        ans = [line.strip('\n') for line in f]
    for i in range(len(lines)):
        l = lines[i].split('\t')
        off = 3
        for j in range(4):
            ai = ans[i].split('\t')[j]
            if ai == 'A':
                ai = 0
            if ai == 'B':
                ai = 1
            if ai == 'C':
                ai = 2
            if ai == 'D':
                ai = 3
            a = l[off + 1 + ai]
            s = ('1 ' + l[2] + ' ' + l[off] + '\t' + a + '\t\t' +
                 l[off + 1] + '|' + l[off + 2] + '|' +
                 l[off + 3] + '|' + l[off + 4])
            off = off + 5
            fout.write(s + '\n')
    fout.close()


def build(opt):
    dpath = os.path.join(opt['datapath'], 'MCTest')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = 'mctest.tar.gz'
        url = 'http://parl.ai/downloads/mctest/' + fname
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)

        dpext = os.path.join(dpath, 'mctest')
        create_fb_format(dpath, 'train160',
                         os.path.join(dpext, 'MCTest', 'mc160.train'), None)
        create_fb_format(dpath, 'valid160',
                         os.path.join(dpext, 'MCTest', 'mc160.dev'), None)
        create_fb_format(dpath, 'test160',
                         os.path.join(dpext, 'MCTest', 'mc160.test'),
                         os.path.join(dpext, 'MCTestAnswers', 'mc160.test.ans'))
        create_fb_format(dpath, 'train500',
                         os.path.join(dpext, 'MCTest', 'mc500.train'), None)
        create_fb_format(dpath, 'valid500',
                         os.path.join(dpext, 'MCTest', 'mc500.dev'), None)
        create_fb_format(dpath, 'test500',
                         os.path.join(dpext, 'MCTest', 'mc500.test'),
                         os.path.join(dpext, 'MCTestAnswers', 'mc500.test.ans'))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
