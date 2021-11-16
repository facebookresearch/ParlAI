#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

ROOT_URL = (
    "https://github.com/google-research-datasets/Taskmaster/raw/master/TM-3-2020/"
)

RESOURCES = [
    # raw data files
    DownloadableFile(
        f"{ROOT_URL}/ontology/apis.json",
        "apis.json",
        "7fe0545dece5dce292c19cc21b02afb0b88fb02a7929d2f8605fe4d6d965a6e0",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_00.json",
        "data_00.json",
        "4edf97557e1aa7f654bf97994b5eae42653ef6d4f5e50136f7664d7e5cdff7b6",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_01.json",
        "data_01.json",
        "a4fef75ec7824bb3fb29b8afe9ed63f3354b81077bb9b6f58f2349bb60d660fc",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_02.json",
        "data_02.json",
        "25cb4788c4c857740152612397b0e687dfeade14c7dbfeed501a53290725aedf",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_03.json",
        "data_03.json",
        "5e1f15544f259c804438458f34a6c7f00400e30dba7648c37a0b31886c5dd903",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_04.json",
        "data_04.json",
        "17d8deb3bc6c3551c564cb9003a47202a835e011477b1e3c81711b083bf4064b",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_05.json",
        "data_05.json",
        "12e58018be7b42aa16968ab914d03c56af9fae9fb5feec7448e01f738c50d2fe",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_06.json",
        "data_06.json",
        "a4e290dafc6362ca8d4d786d6e70f6339297b0414cca4fe0ec79c1d383d65673",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_07.json",
        "data_07.json",
        "becec1c0e9b88781fbb81f5d032826f85d77756cfae7c37acbca9b4b2a470fdb",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_08.json",
        "data_08.json",
        "c2718197cedfb1ff7b830492ddbdc36c213ac98c83ba974ff0ea0eeeffdfe9b5",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_09.json",
        "data_09.json",
        "a8867eb49010fa735b3c0b2f47df4cc225c79b6ca919009db857f454cccc986d",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_10.json",
        "data_10.json",
        "88a8ce81f2c843c7a52a1a6c016d790e6547d5a3ffdd000ca00f0676c02e077b",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_11.json",
        "data_11.json",
        "10e5d2dadb7a55b08d41605498b46320fa2df7b87c71a820df4245e9eeeba4d1",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_12.json",
        "data_12.json",
        "be255d29439d10fc33bb222a1294da0af7ed4bdf8ede84c389d9b744e9cdf343",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_13.json",
        "data_13.json",
        "a477d4668398b86bc7f68f86c20a423e81b4dbc41002a5d605dee0c6cee250b8",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_14.json",
        "data_14.json",
        "92c7a0ca1fce8845610a7fdb00972750228351b2c2065e76120c328c806cc1e5",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_15.json",
        "data_15.json",
        "e67e9d649d29f3edc12796cb4b1308cc7cb89f676c12e96d8fea558510b6ee18",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_16.json",
        "data_16.json",
        "f0d3f22ee8b9224ea4efc57182f1620f849f3b90050cc04992319aab04e7b453",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_17.json",
        "data_17.json",
        "15c469a030aae16a98969675b238f368436ec0a5074f3eb79458fdc81cf489d4",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_18.json",
        "data_18.json",
        "28dfe3f65c1cd7bab8ed9da0e667ee1aec1561cf41237531f30c889cafb0997d",
        zipped=False,
    ),
    DownloadableFile(
        f"{ROOT_URL}/data/data_19.json",
        "data_19.json",
        "7ae99d1237c25519269c3e6969ab0ecd88509828c436cbecb8a86831007c3868",
        zipped=False,
    ),
]


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt["datapath"], "taskmaster-3")
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
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
