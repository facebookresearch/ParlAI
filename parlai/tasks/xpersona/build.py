#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile
import json
from parlai.utils.io import PathManager

RESOURCES = [
    ########En########
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/En_persona_test.json',
        'En_test_tmp.json',
        '8baa09a8064a22967544f501821aa114393a59339c0559da8afa160966ba87c9',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/En_persona_train.json',
        'En_train_tmp.json',
        'e23112bba7320f798b07afb4c5acc3edad2a2ccb7df5cc46f141a0c79ff4665c',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/En_persona_valid.json',
        'En_valid_tmp.json',
        '08ed3d41c5b0681c2d125a5312b43d926a8a5aa1d10a5df655d17f4c56dab635',
        zipped=False,
    ),
    #######Fr########
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Fr_persona_train_corrected.json',
        'Fr_train_tmp.json',
        '40e66e91aa6360eeda642c2c674f03c52854626eaf35da50084d26ff42f61292',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Fr_persona_split_test_human_annotated.json',
        'Fr_test_tmp.json',
        '0783fcf01bdf4c27ec28120a9b23bf4f0248e97ae476e03cd7e759cb2667ab23',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Fr_persona_split_valid_human_annotated.json',
        'Fr_valid_tmp.json',
        '8ad86b05aabfadedba7863828b1cc4fdff0926ebf476121268089ac7ed9af149',
        zipped=False,
    ),
    ########Id########
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Id_persona_train_corrected.json',
        'Id_train_tmp.json',
        'fee1a9769fe707fd09401c33bdf3b3cd4f7b5fd100998577f3f179c42423bc4f',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Id_persona_split_test_human_annotated.json',
        'Id_test_tmp.json',
        'e0bcd3c02f318f4381c42798a2ce6e0a10237b6998793076fbf97f633a9f2563',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Id_persona_split_valid_human_annotated.json',
        'Id_valid_tmp.json',
        '8f70cab662f082ae3ee2abce9e9cac619ebc632a2820d93414a59005dbf75d7e',
        zipped=False,
    ),
    ########It########
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/It_persona_train_corrected.json',
        'It_train_tmp.json',
        '9636893050ad16dc2daabfe8bde6979c9c780c5da9d0790f5668e198aac18b8f',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/It_persona_split_test_human_annotated.json',
        'It_test_tmp.json',
        '690103031791fd5c6763176a074cdb65e249adc784e2d577110c2d9430a02a87',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/It_persona_split_valid_human_annotated.json',
        'It_valid_tmp.json',
        '720dc91d3f9bc6a56ac229c6800cde63c71677a3db8d0ace7a59a7f94d89df3d',
        zipped=False,
    ),
    ########Jp########
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Jp_persona_train_corrected.json',
        'Jp_train_tmp.json',
        '808ee29c79303e1300b38aceee79111ffe1fd2a2facb09a0956615a61d840738',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Jp_persona_split_test_human_annotated.json',
        'Jp_test_tmp.json',
        '682fed16148517097437942088d225bd728cb7b41aa390559681ae73e5e6848f',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Jp_persona_split_valid_human_annotated.json',
        'Jp_valid_tmp.json',
        'a86bb811364d100bc77ddc8038265df1e01bbc3be095e56c6ec8f179e6365d75',
        zipped=False,
    ),
    ########Ko########
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Ko_persona_train_corrected.json',
        'Ko_train_tmp.json',
        '105d6a08d02e76f1d006edb1819b96a8b5fa8d94b3ed278936bcf171368809b7',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Ko_persona_split_test_human_annotated.json',
        'Ko_test_tmp.json',
        'f7ac6bd2aec7014a28d34bb34dceed653b389ce25d21ca770e77142578dc70a6',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Ko_persona_split_valid_human_annotated.json',
        'Ko_valid_tmp.json',
        '188470f863f639946bc8248a9f6aa1e589b41ec61792b88b81e7a95c72deeae0',
        zipped=False,
    ),
    ########Zh########
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Zh_persona_train_corrected.json',
        'Zh_train_tmp.json',
        'e07899fa91edd127ec77502bd604693c40e264b60225976e2ac6ed145d080323',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Zh_persona_split_test_human_annotated.json',
        'Zh_test_tmp.json',
        '0767a4a27c765277792597502f57ea8bb80bf7be94613b0d833107f66a7d3512',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Zh_persona_split_valid_human_annotated.json',
        'Zh_valid_tmp.json',
        'cfa90117d73fe294a1b776b2d1c7b53711bfcd724f5833204726c910dad5482d',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'XPersona')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)
        _create_parlai_format(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)


def _create_parlai_format(dpath: str):
    datatypes = ['train', 'valid', 'test']
    languages = ['En_', 'Zh_', 'Fr_', 'Ko_', 'Id_', 'Jp_', 'It_']
    for language in languages:
        for datatype in datatypes:
            datatype_full = language + datatype + '_tmp'
            datatype_rename = language + datatype
            load_path = os.path.join(dpath, f'{datatype_full}.json')
            save_path = os.path.join(dpath, f'{datatype_rename}.txt')
            with PathManager.open(load_path, 'r', encoding='utf8') as f_read:
                data = json.load(f_read)
            with PathManager.open(save_path, 'w', encoding='utf8') as f_write:
                for content in data:
                    line_num = 0
                    personas = content['persona']
                    dialogs = content['dialogue']
                    for persona in personas:
                        line_num += 1
                        f_write.write(str(line_num) + ' your persona:' + persona + '\n')
                    for utterance_A, utterance_B in dialogs:
                        line_num += 1
                        f_write.write(f"{line_num} {utterance_A}\t{utterance_B}\n")
            os.remove(load_path)
