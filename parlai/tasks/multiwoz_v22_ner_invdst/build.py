#!/usr/bin/env python3

import parlai.core.build_data as build_data
import os

ROOT_URL = (
    "https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2"
)
DATA_LEN = {"train": 17, "dev": 2, "test": 2}


def fold_size(fold):
    return DATA_LEN[fold]


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt["datapath"], "multiwoz_v22")
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
            # filename = 'schema.json'
            # url = f'{ROOT_URL}/{split_type}/{filename}'

            build_data.make_dir(outpath)
            # build_data.download(url, outpath, filename)
            for file_id in range(1, DATA_LEN[split_type] + 1):
                filename = f"dialogues_{file_id:03d}.json"
                url = f"{ROOT_URL}/{split_type}/{filename}"
                build_data.download(url, outpath, filename)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
