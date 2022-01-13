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
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Zh_persona_train_corrected.json',
        'Zh_persona_train_corrected.json',
        'e07899fa91edd127ec77502bd604693c40e264b60225976e2ac6ed145d080323',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Zh_persona_split_test_human_annotated.json',
        'Zh_persona_split_test_human_annotated.json',
        '0767a4a27c765277792597502f57ea8bb80bf7be94613b0d833107f66a7d3512',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/HLTCHKUST/Xpersona/master/dataset/Zh_persona_split_valid_human_annotated.json',
        'Zh_persona_split_valid_human_annotated.json',
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
    """
    Copy data into the format read by ParlAIDialogTeacher.

    'text' will be from the free Turker, who speaks first, and 'label' will be from the
    guided Turker.
    """

    datatypes = {'train':'Zh_persona_train_corrected', 'test':'Zh_persona_split_test_human_annotated', 'valid':'Zh_persona_split_valid_human_annotated'}
    for datatype in datatypes:
        load_path = os.path.join(dpath, f'{datatypes[datatype]}.json')
        save_path = os.path.join(dpath, f'{datatype}.json')

        print(f'Loading {load_path}.')
        with PathManager.open(load_path, 'r', encoding='utf8') as f_read:
            data = json.load(f_read)

        print(f'Saving to {save_path}')
        with PathManager.open(save_path, 'w', encoding='utf8') as f_write:
            n = 0
            for content in data:
                n += 1
                dialog = content['dialogue']
                new_episode = []
                for text in dialog:
                    new_episode.append({
                        'id': 'partner{}'.format(1),
                        'text': "".join(text[0].split(','))
                    })
                    new_episode.append({
                        'id': 'partner{}'.format(2),
                        'text': "".join(text[1].split(','))
                    })
                print(json.dumps({'dialog': [new_episode]}, ensure_ascii=False), file=f_write)
