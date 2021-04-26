#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import random
import shutil
from tqdm import tqdm

import parlai.core.build_data as build_data
from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager
from parlai.utils.logging import logger


RESOURCES = [
    DownloadableFile(
        'https://github.com/festvox/datasets-CMU_DoG/archive/618a14f27546165859305649aa84e6ac8710bb63.zip',
        'cmu_dog.zip',
        'f8ba8820cf86ee1c196b237b0cde80edba940e4ddea28c582830f6d098b3c769',
    )
]

UNZIPPED_PARENT_DIR = 'datasets-CMU_DoG-618a14f27546165859305649aa84e6ac8710bb63'


def build(opt):
    dpath = os.path.join(opt['datapath'], 'cmu_dog')
    version = '1.2'
    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        move_unzipped_files_up(dpath)
        consolidate_wiki_data(dpath)
        consolidate_convos(dpath)
        build_deduped_split(dpath)
        split_into_seen_unseen(dpath)

        build_data.mark_done(dpath, version)


def move_unzipped_files_up(dpath: str):
    unzipped_path = os.path.join(dpath, UNZIPPED_PARENT_DIR)
    for f in os.listdir(unzipped_path):
        shutil.move(os.path.join(unzipped_path, f), dpath)
    shutil.rmtree(unzipped_path)


def consolidate_wiki_data(dpath: str):
    all_articles = {}
    src_dir = os.path.join(dpath, 'WikiData')
    for f_name in tqdm(os.listdir(src_dir)):
        with open(os.path.join(src_dir, f_name)) as f:
            wiki_page = json.load(f)
        idx = wiki_page['wikiDocumentIdx']
        all_articles[idx] = wiki_page
    dest_path = os.path.join(dpath, 'wiki_data.json')
    with open(dest_path, 'w') as d:
        json.dump(all_articles, d, indent=2)
    shutil.rmtree(src_dir)


def consolidate_convos(dpath: str):
    os.makedirs(os.path.join(dpath, 'conversations'), exist_ok=True)
    for split in ['train', 'valid', 'test']:
        consolidate_convo_split(dpath, split)


def consolidate_convo_split(dpath: str, split: str):
    all_convos = {}
    src_dir = os.path.join(dpath, 'Conversations', split)
    for f_name in tqdm(os.listdir(src_dir)):
        with open(os.path.join(src_dir, f_name)) as f:
            convo = json.load(f)
        cid = f_name.split('.')[0]
        all_convos[cid] = convo
    dest_path = os.path.join(dpath, 'conversations', f"{split}.json")
    with open(dest_path, 'w') as dest:
        json.dump(all_convos, dest, indent=2)
    shutil.rmtree(src_dir)


def build_deduped_split(dpath: str):
    """
    Original CMU-DoG has 110 ids that are used in multiple of train/valid/test.

    Get rid of the duplication.
    """
    cdir = os.path.join(dpath, "conversations")
    data = {}
    for fold in ["test", "valid", "train"]:
        fpath = os.path.join(cdir, f"{fold}.json")
        with PathManager.open(fpath) as f:
            data[fold] = json.load(f)

    train_len = len(data["train"])
    valid_len = len(data["valid"])
    test_len = len(data["test"])
    logger.info(
        f"Converation count with duplicates: train-{train_len}, valid-{valid_len}, test-{test_len}"
    )

    train_valid = set(data["train"].keys()) & set(data["valid"].keys())
    train_test = set(data["train"].keys()) & set(data["test"].keys())
    valid_test = set(data["valid"].keys()) & set(data["test"].keys())

    for key in train_valid:
        data["train"].pop(key)
    for key in train_test:
        data["train"].pop(key)
    for key in valid_test:
        data["test"].pop(key)

    train_len = len(data["train"])
    valid_len = len(data["valid"])
    test_len = len(data["test"])
    logger.info(
        f"Converation count without duplicates: train-{train_len}, valid-{valid_len}, test-{test_len}"
    )

    for fold in ["test", "valid", "train"]:
        fpath = os.path.join(cdir, f"{fold}_deduped.json")
        with PathManager.open(fpath, "w+") as f:
            json.dump(data[fold], f, indent=2)


def split_into_seen_unseen(dpath: str):
    """
    Following WoW, we have overlap in train, valid, and test seen but none in test
    unseen. Do an 80:10:5:5 split between train, valid, test_seen, test_unseen or as
    close to it.

    ~205 documents for test_unseen to do this, and movies 1 and 3 have 90 and 117
    movies, respectively, which is about that
    """
    random.seed(42)
    cdir = os.path.join(dpath, "conversations")
    new = {"train": {}, "valid": {}, "test_seen": {}, "test_unseen": {}}
    for fold in ["test", "valid", "train"]:
        with PathManager.open(os.path.join(cdir, f"{fold}_deduped.json")) as f:
            data = json.load(f)
        for k, v in data.items():
            if v["wikiDocumentIdx"] == 1 or v["wikiDocumentIdx"] == 3:
                new["test_unseen"][k] = v
            else:
                rand = random.randint(1, 95)
                if rand <= 80:
                    new["train"][k] = v
                elif rand <= 90:
                    new["valid"][k] = v
                else:
                    new["test_seen"][k] = v

    for fold in new:
        with PathManager.open(
            os.path.join(cdir, f"{fold}_split_seen_unseen.json"), "w+"
        ) as f:
            json.dump(new[fold], f, indent=2)
        c_cnt = len(new[fold])
        logger.info(f"Seen/unseen {fold} conversation count: {c_cnt}")
