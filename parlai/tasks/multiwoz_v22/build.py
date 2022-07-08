#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

# Pin against a specific commit since folks occasionally post fixes
MULTIWOZ_URL_BASE = "https://raw.githubusercontent.com/budzianowski/multiwoz/01e689362833ce33427a771a21cefe253e8f5886/"

MULTIWOZ_22_URL_BASE = MULTIWOZ_URL_BASE + "data/MultiWOZ_2.2/"

RESOURCES = [
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'dialog_acts.json',
        'dialog_acts.json',
        '328f392165e7826db9f827731b14b5cc04e79e9e3c6332bfb192a1ea17f8e9b6',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'schema.json',
        'schema.json',
        'ae9e2390f38fb967af64623c2f4f7e0c636fb377ad523b582a03161d3ddbdf68',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'dev/dialogues_001.json',
        'dev/dialogues_001.json',
        'e7ddb563e4da5766ea820cc826dead77e7ca219c19b761e218d62d9c999a252e',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'dev/dialogues_002.json',
        'dev/dialogues_002.json',
        'ede6a2c17fd6c5846214b8cabc1ef8f7cc8be01cfbacaa162bcafec9e87724e9',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'test/dialogues_001.json',
        'test/dialogues_001.json',
        'd6f43876cf130fdb2dfa8f96bc056b0995354137f02122e004925d01264ed386',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'test/dialogues_002.json',
        'test/dialogues_002.json',
        '89af95d8f596a448e733d59b31be78f1dd1632eddd99d5cb298a3fcb1ac9d185',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_001.json',
        'train/dialogues_001.json',
        '895a8109bf01fa5ecf15ccdbd2dfe1628bd923f6b61dcd2e26b10ee5076a1596',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_002.json',
        'train/dialogues_002.json',
        '2f3ea771d4e01cb2780357738cff7f7496b87d34c221cc240df74501312438d3',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_003.json',
        'train/dialogues_003.json',
        'da24961d28486be2d8462ee4d86a809db819d588ba90ae1a783383d95eb85daa',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_004.json',
        'train/dialogues_004.json',
        '30c1172db1071c853b16215d1946de908d68d2b6ff8da7801de307000a179106',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_005.json',
        'train/dialogues_005.json',
        'eaf58716df5de99524b3e0e7edf477b74749512788a6a51f53f2bdd76768d39a',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_006.json',
        'train/dialogues_006.json',
        '8e75fd543b1964bc5e7118085d977f479c98fcdf6d606b971f67a45fb1745c83',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_007.json',
        'train/dialogues_007.json',
        '02323f8298439d713c6d7d226f4bd7ec246ec993ee11911b54f98cb8a598f206',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_008.json',
        'train/dialogues_008.json',
        '1129fbed480352ae304f0ae5b4915c194e9619c43f1577ccb0d450e10d24ea98',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_009.json',
        'train/dialogues_009.json',
        '87d9e43b1ba51a4a4688703da79d3a09b14d8013570545da24c597daa18e2f45',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_010.json',
        'train/dialogues_010.json',
        'e7ad0d5da2909b08197295e45fe4695b9dc2f67d458374b2aab8db5094e97b26',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_011.json',
        'train/dialogues_011.json',
        '82e2d2900a037b866a01d05974734dd419e89b329fe29ef93b35eea96d27feb8',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_012.json',
        'train/dialogues_012.json',
        'b6bf292325db67682dd7b6fafbf1051cc2262e92f7c37cab213e975888594bb2',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_013.json',
        'train/dialogues_013.json',
        'c33fe4b3952c016e1e1645f472a7097f93dfb476a19940fd9386865ef9adf685',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_014.json',
        'train/dialogues_014.json',
        'ce33dbbf93a40d0dcc671a9d6a2ed1f987914f5b5f05f6df493a5b342efda954',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_015.json',
        'train/dialogues_015.json',
        'd895c0439bc2ad89ef1689896e3be630eadebc33dae77b42a426fd16a271718e',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_016.json',
        'train/dialogues_016.json',
        '3e6bc0bca4262022ccbce0d5ce3150e536e7d21aeb9fbdef9e83cced4dfd124b',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_22_URL_BASE + 'train/dialogues_017.json',
        'train/dialogues_017.json',
        'b6ab2cd9b6e8983364526845b7cbec1749338209bf7ac2313c25e1e2226ebab5',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_URL_BASE + 'db/attraction_db.json',
        'db/attraction_db.json',
        '2aacc620af4025f1eada5ec83057a866f8e8b72b529f71d2f8bf93bcdd8f8751',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_URL_BASE + 'db/bus_db.json',
        'db/bus_db.json',
        '4818e735bae20690f6d0d06bb2ae8eec1981c0b680258a970dc01c9073f3fec9',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_URL_BASE + 'db/hospital_db.json',
        'db/hospital_db.json',
        'f28738bda15e750be912d653c5e68b06af41dba68d9cfa3febfdcfe972e14366',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_URL_BASE + 'db/hotel_db.json',
        'db/hotel_db.json',
        '972bbd65beada7c64f0b87322c694fd9173b46cf8e61ca3bbe951717ac1d1662',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_URL_BASE + 'db/police_db.json',
        'db/police_db.json',
        'd9c2b200fa2dd61b04ce2fe520b0b79dfa68d8b895806cfec6e8a8d3cffa9193',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_URL_BASE + 'db/restaurant_db.json',
        'db/restaurant_db.json',
        '7864b4e36027af0907d6afe9ed962fecd6c6966cd626b8ae9341708a04ea201a',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_URL_BASE + 'db/taxi_db.json',
        'db/taxi_db.json',
        '08b8fb2436abec6d1fe9087054f943d2b31e1c4adc74d8202e4e2149a5630be3',
        zipped=False,
    ),
    DownloadableFile(
        MULTIWOZ_URL_BASE + 'db/train_db.json',
        'db/train_db.json',
        '4818e735bae20690f6d0d06bb2ae8eec1981c0b680258a970dc01c9073f3fec9',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'multiwoz_v22')
    version = '1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        build_data.make_dir(dpath + "/dev")
        build_data.make_dir(dpath + "/test")
        build_data.make_dir(dpath + "/train")

        build_data.make_dir(dpath + "/db")

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        build_data.mark_done(dpath, version_string=version)
