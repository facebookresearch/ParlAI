#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.core.build_data as build_data
import os
from hashlib import sha1
from parlai.tasks.light_dialog_wild.builder import build_from_dump
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/light_project/wild_chats/contents.txt',
        'current_chats.txt',
        'c708fe62692f239a2b35025d71722c7607f863ffa110aa118f2e1d0fa7db4730',
        zipped=False,
    )
]


def download(opt):
    version = 'v0.1'
    # download all of the currently available jsons
    dpath = os.path.join(opt['datapath'], 'light_dialogue_wild', 'dumps')
    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the manifest.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Read the manifest
        with open(os.path.join(dpath, 'current_chats.txt'), 'r') as manifest:
            additional_files = manifest.readlines()

        # Download the files listed in the manifest
        more_resources = []
        for sha_and_filename in additional_files:
            file_hash, filename = sha_and_filename.strip().split('  ')
            more_resources.append(
                DownloadableFile(
                    f'http://parl.ai/downloads/light_project/wild_chats/{filename}',
                    filename,
                    file_hash,
                    zipped=False,
                )
            )

        for downloadable_file in more_resources:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version)

    return dpath, version


def get_fpath(opt):
    fields = [
        'taskname',
        'setting',
        'person_names',
        'persona',
        'emote',
        'speech',
        'action',
        'affordances',
        'repeat',
        'cands',
        'current_self_output',
        'clip_cands',
        'speech_prefix',
        'person_names_prefix',
        'score_cutoff',
        'hard_score_cutoff',
        'max_score_cutoff',
        'date_cutoff',
        'continue_type',
        'unseen_test',
    ]
    fpath = ''
    for f in fields:
        fpath += f + str(opt['light_use_' + f]) + "_"
    if opt.get('light_model_name'):
        fpath += f"model_name{opt['light_model_name']}_"
    return str(sha1(fpath[:-1].encode('utf-8')).hexdigest())


def build(opt):
    dpath, version = download(opt)
    if 'light_use_speech_prefix' not in opt:
        opt['light_use_speech_prefix'] = True
    # create particular instance of dataset depending on flags..
    fpath = get_fpath(opt)
    dump_path = os.path.join(opt['datapath'], 'light_dialogue_wild', 'dumps')
    data_path = os.path.join(opt['datapath'], 'light_dialogue_wild', fpath)
    if not build_data.built(data_path, version):
        if build_data.built(data_path):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(data_path)
        build_data.make_dir(data_path)
        build_from_dump(opt, data_path, dump_path)
        # Mark the data as built.
        build_data.mark_done(data_path, version)
