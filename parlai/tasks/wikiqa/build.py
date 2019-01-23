#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def create_fb_format(outpath, dtype, inpath):
    print('building fbformat:' + dtype)
    fout = open(os.path.join(outpath, dtype + '.txt'), 'w')
    with open(inpath) as f:
        lines = [line.strip('\n') for line in f]
    lastqid, lq, ans, cands = None, None, None, None
    for i in range(2, len(lines)):
        l = lines[i].split('\t')
        lqid = l[0]  # question id
        if lqid != lastqid:
            if lastqid is not None:
                # save
                s = '1 ' + lq + '\t' + ans.lstrip('|') + '\t\t' + cands.lstrip('|')
                if (dtype.find('filtered') == -1) or ans != '':
                    fout.write(s + '\n')
            # reset
            cands = ''
            ans = ''
            lastqid = lqid
        lcand = l[5]  # candidate answer / sentence from doc
        lq = l[1]  # question
        llabel = l[6]  # 0 or 1
        if int(llabel) == 1:
            ans = ans + '|' + lcand
        cands = cands + '|' + lcand
    fout.close()


def build(opt):
    dpath = os.path.join(opt['datapath'], 'WikiQA')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = 'wikiqa.tar.gz'
        url = 'http://parl.ai/downloads/wikiqa/' + fname
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)

        dpext = os.path.join(dpath, 'WikiQACorpus')
        create_fb_format(dpath, 'train',
                         os.path.join(dpext, 'WikiQA-train.tsv'))
        create_fb_format(dpath, 'valid',
                         os.path.join(dpext, 'WikiQA-dev.tsv'))
        create_fb_format(dpath, 'test',
                         os.path.join(dpext, 'WikiQA-test.tsv'))
        create_fb_format(dpath, 'train-filtered',
                         os.path.join(dpext, 'WikiQA-train.tsv'))
        create_fb_format(dpath, 'valid-filtered',
                         os.path.join(dpext, 'WikiQA-dev.tsv'))
        create_fb_format(dpath, 'test-filtered',
                         os.path.join(dpext, 'WikiQA-test.tsv'))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
