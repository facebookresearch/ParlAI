#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data


def create_fb_format(outpath, dtype, inpath):
    print("building fbformat:" + dtype)
    fout = open(outpath + dtype + '.txt', 'w')
    with open(inpath) as f:
        lines = [line.strip('\n') for line in f]
    lastqid = None
    for i in range(2, len(lines)):
        l = lines[i].split('\t')
        lqid = l[0]  # question id
        if lqid != lastqid:
            if lastqid is not None:
                # save
                s = '1 ' + lq + '\t' + ans.lstrip('|') + '\t' + cands.lstrip('|')
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
    dpath = opt['datapath'] + "/WikiQA/"

    if not build_data.built(dpath):
        print("[building data: " + dpath + "]")
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = "wikiqa.tar.gz"
        url = "https://s3.amazonaws.com/fair-data/parlai/wikiqa/" + fname
        build_data.download(dpath, url)
        build_data.untar(dpath, fname)

        dpext = dpath + 'WikiQACorpus/'
        create_fb_format(dpath, 'train', dpext + 'WikiQA-train.tsv')
        create_fb_format(dpath, 'valid', dpext + 'WikiQA-dev.tsv')
        create_fb_format(dpath, 'test', dpext + 'WikiQA-test.tsv')
        create_fb_format(dpath, 'train-filtered', dpext + 'WikiQA-train.tsv')
        create_fb_format(dpath, 'valid-filtered', dpext + 'WikiQA-dev.tsv')
        create_fb_format(dpath, 'test-filtered', dpext + 'WikiQA-test.tsv')

        # Mark the data as built.
        build_data.mark_done(dpath)
