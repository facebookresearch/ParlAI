#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Included as a convenience in case the files get updated.
"""


import hashlib
import wget

MULTIWOZ_URL_BASE = "https://raw.githubusercontent.com/budzianowski/multiwoz/01e689362833ce33427a771a21cefe253e8f5886/"
MULTIWOZ_22_URL_BASE = MULTIWOZ_URL_BASE + "/data/MultiWOZ_2.2/"

WANT = [
    "dialog_acts.json",
    "schema.json",
    "dev/dialogues_001.json",
    "dev/dialogues_002.json",
    "test/dialogues_001.json",
    "test/dialogues_002.json",
    "train/dialogues_001.json",
    "train/dialogues_002.json",
    "train/dialogues_003.json",
    "train/dialogues_004.json",
    "train/dialogues_005.json",
    "train/dialogues_006.json",
    "train/dialogues_007.json",
    "train/dialogues_008.json",
    "train/dialogues_009.json",
    "train/dialogues_010.json",
    "train/dialogues_011.json",
    "train/dialogues_012.json",
    "train/dialogues_013.json",
    "train/dialogues_014.json",
    "train/dialogues_015.json",
    "train/dialogues_016.json",
    "train/dialogues_017.json",
]

FILES = [(x, MULTIWOZ_22_URL_BASE + x, "MULTIWOZ_22_URL_BASE") for x in WANT]

WANT = [
    "db/attraction_db.json",
    "db/bus_db.json",
    "db/hospital_db.json",
    "db/hotel_db.json",
    "db/police_db.json",
    "db/restaurant_db.json",
    "db/taxi_db.json",
    "db/train_db.json",
]

FILES += [(x, MULTIWOZ_URL_BASE + x, "MULTIWOZ_URL_BASE") for x in WANT]


def checksum(dpath):
    """
    Checksum on a given file.

    :param dpath: path to the downloaded file.
    """
    sha256_hash = hashlib.sha256()
    with open(dpath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


for f in FILES:
    name, path, start = f
    print("     DownloadableFile(")
    print(f"        {start} + '{name}',")
    print(f"        '{name}',")
    filename = wget.download(path, bar=None)
    print(f"        '{checksum(filename)}',")
    print("        zipped = False,")
    print("    ),")
