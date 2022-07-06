#!/usr/bin/env python3


# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
import gzip
import json
from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager

RESOURCES = [
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Appliances.json.gz',
        'qa_Appliances.json.gz',
        '9c613a5dfedd1071431faa29de903b1b0e592c5ac1c7861c26d8b69dfda8ac78',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Arts_Crafts_and_Sewing.json.gz',
        'qa_Arts_Crafts_and_Sewing.json.gz',
        'c9aad6d615294571c1be7ea6a88730829a68e701ca7d1168f4d6b5234c37ac65',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Automotive.json.gz',
        'qa_Automotive.json.gz',
        'ca2da4b9d3afd3e6c915d69b34618bdcf9c6febadd7389f368fb51e9e1585009',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Baby.json.gz',
        'qa_Baby.json.gz',
        'acca09d58e7b5487c41de1e21f5a4c676f9aaf1b0534cbeb73c2d5a7444231a0',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Beauty.json.gz',
        'qa_Beauty.json.gz',
        'e4e7d28e917fd1b0a605c85eff4c5d3e832a925ab6ed1216147b306984457b90',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Cell_Phones_and_Accessories.json.gz',
        'qa_Cell_Phones_and_Accessories.json.gz',
        '2ca990b05fbf1282aeeaab139ca8ad005cd540c0ad198ae2558d43cb01f24b42',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Clothing_Shoes_and_Jewelry.json.gz',
        'qa_Clothing_Shoes_and_Jewelry.json.gz',
        '2c8e24b6d89ccc4084c694a6197d862cdfdc90224877e566dc8170c86ba38d68',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Electronics.json.gz',
        'qa_Electronics.json.gz',
        '4ead1073043b6103a3a8eb738a04143f448cd3e200df074ad5071a24ad6bae95',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Grocery_and_Gourmet_Food.json.gz',
        'qa_Grocery_and_Gourmet_Food.json.gz',
        '438a9b97924098484c73d59e9f47e6bdb7898fdc94ae5fcfead0493a37443817',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Health_and_Personal_Care.json.gz',
        'qa_Health_and_Personal_Care.json.gz',
        '5f3564cf88a56fdf6913ca8e0e7e51478ef829b8faff6bec0e33f46e567bcbf6',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Home_and_Kitchen.json.gz',
        'qa_Home_and_Kitchen.json.gz',
        'eefa1f213bb476478c40ab9c2081c7dd0925f2f1df5d3efeb43f6e5715daaeaf',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Industrial_and_Scientific.json.gz',
        'qa_Industrial_and_Scientific.json.gz',
        '9ae1e9c2db859061350559d59c44125131d8af1149d6e99f45a73687f67a6b0e',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Musical_Instruments.json.gz',
        'qa_Musical_Instruments.json.gz',
        'a677605d6b6e612a485e939855f85c51a3bf1678da56c81ad5799ce1ef70d8bf',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Office_Products.json.gz',
        'qa_Office_Products.json.gz',
        '78e895f0df8e647d1c1a0a46f112f57adad9c49584dd48fb90e14648da790025',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Patio_Lawn_and_Garden.json.gz',
        'qa_Patio_Lawn_and_Garden.json.gz',
        '30a7bc60b39345a818a91a87f528afd860dc479d97115c4274056219471f5222',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Pet_Supplies.json.gz',
        'qa_Pet_Supplies.json.gz',
        '13a6a722f5f1c3209a0296ef87115ad2cc3ffa490b098c278596a0994484ef41',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Software.json.gz',
        'qa_Software.json.gz',
        '54847f45ed575611bc3abc88f176491678d1f0c38cad3a15ad5462f99c30e31c',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Sports_and_Outdoors.json.gz',
        'qa_Sports_and_Outdoors.json.gz',
        'a513460612695249027345f6cb8bde35d8dca664927582f4a1c429a45e60b62d',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Tools_and_Home_Improvement.json.gz',
        'qa_Tools_and_Home_Improvement.json.gz',
        'fe29b60e01c9867e7f62ea416bee300cd68f6c7230ac518f94f2fb252e322bc8',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Toys_and_Games.json.gz',
        'qa_Toys_and_Games.json.gz',
        '231582d4d5aa4959b6344d21ea0bf1b4872aa59645bb9e8107d1056922673dfe',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Video_Games.json.gz',
        'qa_Video_Games.json.gz',
        '45d626d00f393e9df0191d12e12fbdce6b6655ad14715bba2f8d9fe58fc7e1c2',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Automotive.json.gz',
        'QA2_Automotive.json.gz',
        '2a8209571736a56cfef5a3029545b614b3adfae1de166c179a2488f42fac2533',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Beauty.json.gz',
        'QA2_Beauty.json.gz',
        '1ac51134997c86cc59380b7f1be1885010dd1dd9e0aac7a503255c4f68bda7d0',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Cell_Phones_and_Accessories.json.gz',
        'QA2_Cell_Phones_and_Accessories.json.gz',
        '818a0cad91543a508f90ed0b3db11e71c964036a77c896a41cc7aa79283f1940',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Clothing_Shoes_and_Jewelry.json.gz',
        'QA2_Clothing_Shoes_and_Jewelry.json.gz',
        '47da83867a89fa7ab376040dc82944e74935cc81e1b86a6e3fc053a7a2dfdb32',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Electronics.json.gz',
        'QA2_Electronics.json.gz',
        'ab1081838b2f603708c3e6b9983a63f65512f46f1e1e0463f4437c88da0c7663',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Grocery_and_Gourmet_Food.json.gz',
        'QA2_Grocery_and_Gourmet_Food.json.gz',
        '140cdacf2b9bbb27bac0c60dc8a92f1c211cead68879451fac73c5e424bc1dd2',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Health_and_Personal_Care.json.gz',
        'QA2_Health_and_Personal_Care.json.gz',
        'c14dcd0bf837b3a26463de3cb2db8c3ad81bb262377ddc36610e8b674d49c833',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Home_and_Kitchen.json.gz',
        'QA2_Home_and_Kitchen.json.gz',
        '2822abbc06d24aa248c01b0a7edd982c8421c8eabaf5af421ded8eadbaf2dbfd',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Musical_Instruments.json.gz',
        'QA2_Musical_Instruments.json.gz',
        'e550e0f6a00db32966cb798ee6f3754cf8c16b474ae9ce3b0b8621faa45b4e15',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Office_Products.json.gz',
        'QA2_Office_Products.json.gz',
        '1ecb9cd8e7ad0f9049b2b0c350ce7eb304a32e685472da1d7da74dd8a5cb411f',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Patio_Lawn_and_Garden.json.gz',
        'QA2_Patio_Lawn_and_Garden.json.gz',
        'f78d127dbb9499af5f87ded005747061cc44c1114edeec8cae7346e044c697d3',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Pet_Supplies.json.gz',
        'QA2_Pet_Supplies.json.gz',
        'c0f31cb041a6bbc7dc5f9006c41458f8d9120c949fb6f1e6fe753cec4353e0f1',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Sports_and_Outdoors.json.gz',
        'QA2_Sports_and_Outdoors.json.gz',
        '521320c8162c17b361b070fe5f3fc8db4c4205508eff757f12fce284cbd555eb',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Tools_and_Home_Improvement.json.gz',
        'QA2_Tools_and_Home_Improvement.json.gz',
        '473ba7d623b8c43d832a9d7a3440796edcb4320211ca5b7c5dab69ed56b66a50',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Toys_and_Games.json.gz',
        'QA2_Toys_and_Games.json.gz',
        'dbeeaef459d266f383fea2845c8c76da6737812724abfb335db6a294870b1b10',
        zipped=False,
    ),
    DownloadableFile(
        'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Video_Games.json.gz',
        'QA2_Video_Games.json.gz',
        'c54bb576b136146a3afa2a80033312b8b5ee6db721918db9875622e348a7c11c',
        zipped=False,
    ),
]


def parse_gzip(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))


def build(opt):
    dpath = os.path.join(opt['datapath'], 'AmazonQA')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)
            new_filename = downloadable_file.file_name[:-3]
            print('[ unpacking data: ' + downloadable_file.file_name + ' ]')
            f = open(dpath + '/' + new_filename, 'w')
            for l in parse_gzip(dpath + '/' + downloadable_file.file_name):
                f.write(l + '\n')
            PathManager.rm(dpath + '/' + downloadable_file.file_name)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
