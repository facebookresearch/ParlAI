# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.
import gzip
import json
import os

import parlai.core.build_data as build_data


def read_gz(filename, delete_gz=True):
    f = gzip.open(filename, 'rb')
    lines = [x.decode('utf-8') for x in f.readlines()]
    if delete_gz:
        os.remove(filename)
    return lines


def create_fb_format(outpath, dtype, inpath):
    print('building fbformat:' + dtype)

    lines = read_gz(inpath)

    # save the raw json version for span selection task (default)
    fout1 = open(os.path.join(outpath, dtype + '.txt'), 'w')
    for line in lines:
        fout1.write(line.rstrip("\n") + "\n")
    fout1.close()

    # save the file for passage selection task
    fout2 = open(os.path.join(outpath, dtype + '.passage.txt'), 'w')
    for line in lines:
        dic = json.loads(line)
        lq = dic["query"]
        if dtype != "test":
            ans = "|".join([d["passage_text"] for d in dic["passages"] if d["is_selected"] == 1])
            cands = "|".join([d["passage_text"] for d in dic["passages"] if d["is_selected"] == 0])
            cands = ans + "|" + cands
            if ans == "": continue  # if no true label, skip for now
        else:  # ground truth for test data is not available yet
            ans = ""
            cands = "|".join([d["passage_text"] for d in dic["passages"]])
        s = '1 ' + lq + '\t' + ans.lstrip("|") + '\t\t' + cands
        fout2.write(s + '\n')
    fout2.close()


def build(opt):
    dpath = os.path.join(opt['datapath'], 'MS_MARCO')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data
        url = "https://msmarco.blob.core.windows.net/msmarco/"

        fname = "train_v1.1.json.gz"
        build_data.download(url + fname, dpath, 'train.gz')

        fname = "dev_v1.1.json.gz"
        build_data.download(url + fname, dpath, 'valid.gz')

        fname = "test_public_v1.1.json.gz"
        build_data.download(url + fname, dpath, 'test.gz')

        create_fb_format(dpath, "train", os.path.join(dpath, 'train.gz'))
        create_fb_format(dpath, "valid", os.path.join(dpath, 'valid.gz'))
        create_fb_format(dpath, "test", os.path.join(dpath, 'test.gz'))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
