#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import parlai.core.build_data as build_data
import os

ROOT_URL = "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/raw/master"
DATA_LEN = {"train": 127, "dev": 20, "test": 34}


def fold_size(fold):
    return DATA_LEN[fold]


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt["datapath"], "google_sgd")
    # define version if any
    version = "1.0"

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print("[building data: " + dpath + "]")

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for split_type in ["train", "dev", "test"]:
            outpath = os.path.join(dpath, split_type)
            filename = "schema.json"
            url = f"{ROOT_URL}/{split_type}/{filename}"

            build_data.make_dir(outpath)
            build_data.download(url, outpath, filename)
            for file_id in range(1, DATA_LEN[split_type] + 1):
                filename = f"dialogues_{file_id:03d}.json"
                url = f"{ROOT_URL}/{split_type}/{filename}"
                build_data.download(url, outpath, filename)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
