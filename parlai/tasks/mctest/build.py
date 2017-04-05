#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data


def create_fb_format(outpath, dtype, inpath, inpath2):
    print("building fbformat:" + dtype)
    fout = open(outpath + dtype + '.txt', 'w')
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
            s = ('1 ' + l[2] + ' ' + l[off] + '\t' + a + '\t1\t' +
                 l[off + 1] + '|' + l[off + 2] + '|' +
                 l[off + 3] + '|' + l[off + 4])
            off = off + 5
            fout.write(s + '\n')
    fout.close()

def build(opt):
    dpath = opt['datapath'] + "/MCTest/"

    if not build_data.built(dpath):
        print("[building data: " + dpath + "]")
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = "mctest.tar.gz"
        url = "https://s3.amazonaws.com/fair-data/parlai/mctest/" + fname
        build_data.download(dpath, url)
        build_data.untar(dpath, fname)

        dpext = dpath + 'mctest/'
        create_fb_format(dpath, 'train160', dpext + 'MCTest/mc160.train', None)
        create_fb_format(dpath, 'valid160', dpext + 'MCTest/mc160.dev', None)
        create_fb_format(dpath, 'test160', dpext + 'MCTest/mc160.test',
                         dpext + 'MCTestAnswers/mc160.test.ans')
        create_fb_format(dpath, 'train500', dpext + 'MCTest/mc500.train', None)
        create_fb_format(dpath, 'valid500', dpext + 'MCTest/mc500.dev', None)
        create_fb_format(dpath, 'test500',  dpext + 'MCTest/mc500.test',
                         dpext + 'MCTestAnswers/mc500.test.ans')

        # Mark the data as built.
        build_data.mark_done(dpath)
