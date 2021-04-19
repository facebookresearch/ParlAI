#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Reformats the original CMU_DoG data into fewer files.

You shouldn't need to run this; it's kept here for reference and reproducibility.

Usage: python consolidate.py
Assumes you've already cloned the original dataset into your home directory.
"""

import getpass
import json
import os
import tqdm

USER = getpass.getuser()
SOURCE_DIR = f"/private/home/{USER}/datasets-CMU_DoG/"
OUTDIR = f"/private/home/{USER}/data/cmu_dog/"


def consolidate_wiki_data():
    all_articles = {}
    src_dir = os.path.join(SOURCE_DIR, 'WikiData')
    for f_name in tqdm(os.listdir(src_dir)):
        with open(os.path.join(src_dir, f_name)) as f:
            wiki_page = json.load(f)
        idx = wiki_page['wikiDocumentIdx']
        all_articles[idx] = wiki_page
    dest_path = os.path.join(OUTDIR, 'wiki_data.json')
    with open(dest_path, 'w') as d:
        json.dump(all_articles, d, indent=2)


def consolidate_convos(split):
    all_convos = {}
    src_dir = os.path.join(SOURCE_DIR, 'Conversations', split)
    for f_name in tqdm(os.listdir(src_dir)):
        with open(os.path.join(src_dir, f_name)) as f:
            convo = json.load(f)
        cid = f_name.split('.')[0]
        all_convos[cid] = convo
    dest_path = os.path.join(OUTDIR, 'conversations', f"{split}.json")
    with open(dest_path, 'w') as dest:
        json.dump(all_convos, dest, indent=2)


if __name__ == '__main__':
    consolidate_wiki_data()
    for split in ['train', 'valid', 'test']:
        consolidate_convos(split)
