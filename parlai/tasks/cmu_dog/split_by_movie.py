#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Used to create the seen/unseen split.

You shouldn't need to run this; it's kept here for reference and reproducibility.
"""

import getpass
import json
import os
import random

from parlai.utils.io import PathManager

USER = getpass.getuser()
BASE_DIR = f"/private/home/{USER}/ParlAI/data/cmu_dog/"


def split_into_seen_unseen():
    """
    Following WoW, we have overlap in train, valid, and test seen but none in test
    valid. Do an 80:10:5:5 split between train, valid, test_seen, test_unseen or as
    close to it.

    ~205 documents for test_unseen to do this, and movies 1 and 3 have 90 and 117
    movies, respectively, which is about that
    """
    random.seed(42)
    new = {"train": {}, "valid": {}, "test_seen": {}, "test_unseen": {}}
    for fold in ["test", "valid", "train"]:
        with PathManager.open(os.path.join(BASE_DIR, f"{fold}.json")) as f:
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
            os.path.join(BASE_DIR, f"{fold}_split_seen_unseen.json"), "w+"
        ) as f:
            json.dump(new[fold], f, indent=4)
        print(len(new[fold]))


if __name__ == '__main__':
    split_into_seen_unseen()
