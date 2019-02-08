#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import tqdm
import parlai.core.build_data as build_data


def read_file(filename):
    with open(filename) as f:
        lines = [x for x in f.readlines()]
    return lines


def convert_file(input_file_path):
    print("Reading gzip file {}.".format(input_file_path))
    with gzip.open(input_file_path) as f:
        records = json.load(f)
    n = len(records["passages"].keys())
    for i in tqdm.tqdm(range(n), "Converting"):
        newline_dict = {}
        index = str(i)
        if "test" not in input_file_path:
            newline_dict["answers"] = records["answers"][index]
            newline_dict["wellFormedAnswers"] = records["wellFormedAnswers"][index]
        newline_dict["passages"] = records["passages"][index]
        newline_dict["query"] = records["query"][index]
        newline_dict["query_id"] = records["query_id"][index]
        newline_dict["query_type"] = records["query_type"][index]
        yield newline_dict


def create_fb_format(outpath, dtype, inpath):
    print('building fbformat:' + dtype)
    episodes = list(convert_file(inpath))

    # save the raw json version for span selection task (default)
    with open(os.path.join(outpath, dtype + '.txt'), 'w') as fout1:
        for ep in episodes:
            fout1.write(json.dumps(ep) + "\n")

    # save the file for passage selection task
    with open(os.path.join(outpath, dtype + '.passage.txt'), 'w') as fout2:
        for dic in episodes:
            lq = dic["query"]
            if dtype != "test":
                ans = "|".join([
                    d["passage_text"] for d in dic["passages"] if d["is_selected"] == 1
                ])
                cands = "|".join([
                    d["passage_text"] for d in dic["passages"] if d["is_selected"] == 0
                ])
                cands = ans + "|" + cands
                if ans == "":
                    # if no true label, skip for now
                    continue
            else:
                # ground truth for test data is not available yet
                ans = ""
                cands = "|".join([d["passage_text"] for d in dic["passages"]])
            s = '1 ' + lq + '\t' + ans.lstrip("|") + '\t\t' + cands
            fout2.write(s + '\n')

# Download and build the data if it does not exist.
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

        fname = "train_v2.1.json.gz"
        build_data.download(url + fname, dpath, 'train.gz')

        fname = "dev_v2.1.json.gz"
        build_data.download(url + fname, dpath, 'valid.gz')

        fname = "eval_v2.1_public.json.gz"
        build_data.download(url + fname, dpath, 'test.gz')

        create_fb_format(dpath, "train", os.path.join(dpath, 'train.gz'))
        # os.remove(os.path.join(dpath, 'train.gz'))
        create_fb_format(dpath, "valid", os.path.join(dpath, 'valid.gz'))
        # os.remove(os.path.join(dpath, 'valid.gz'))
        create_fb_format(dpath, "test", os.path.join(dpath, 'test.gz'))
        # os.remove(os.path.join(dpath, 'test.gz'))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
