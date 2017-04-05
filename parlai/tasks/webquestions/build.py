#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
#
# Download and build the data if it does not exist.

import re
import json
import parlai.core.build_data as build_data


def parse_ans(a):
    a = a.lstrip('(list')
    ans = ''
    for a in re.split('\(description', a):
        a = a.strip(' ()"()')
        ans = ans + '|' + a
    return ans.lstrip('|')

def create_fb_format(outpath, dtype, inpath):
    print("building fbformat:" + dtype)
    with open(inpath) as data_file:
        data = json.load(data_file)
    fout = open(outpath + dtype + '.txt', 'w')
    for i in range(len(data)):
        q = data[i]['utterance']
        a = parse_ans(data[i]['targetValue'])
        s = '1 ' + q + '\t' + a
        fout.write(s + '\n')
    fout.close()


def build(opt):
    dpath = opt['datapath'] + "/WebQuestions/"

    if not build_data.built(dpath):
        print("[building data: " + dpath + "]")
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

         # Download the data.
        url = ("https://worksheets.codalab.org/rest/bundles/" +
               "0x4a763f8cde224c2da592b75f29e2f5c2/contents/blob/")
        build_data.download(dpath, url)
        build_data.move(dpath + 'index.html', dpath + 'train.json')

        url = ("https://worksheets.codalab.org/rest/bundles/" +
               "0xe7bac352fce7448c9ef238fb0a297ec2/contents/blob/")
        build_data.download(dpath, url)
        build_data.move(dpath + 'index.html', dpath + 'test.json')

        create_fb_format(dpath, 'train', dpath + 'train.json')
        create_fb_format(dpath, 'valid', dpath + 'train.json')
        create_fb_format(dpath, 'test', dpath + 'test.json')

        # Mark the data as built.
        build_data.mark_done(dpath)
