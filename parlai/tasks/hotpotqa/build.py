#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import json

VERSION = '1'
TRAIN_FILENAME = 'hotpot_train_v{}.1.json'.format(VERSION)
DEV_DISTRACTOR_FILENAME = 'hotpot_dev_distractor_v{}.json'.format(VERSION)
DEV_FULLWIKI_FILENAME = 'hotpot_dev_fullwiki_v{}.json'.format(VERSION)

URL = 'http://curtis.ml.cmu.edu/datasets/hotpot/'
FILE_NAMES = [
    'hotpot_train_v{}.1.json'.format(VERSION),
    'hotpot_dev_distractor_v{}.json'.format(VERSION),
    'hotpot_dev_fullwiki_v{}.json'.format(VERSION),
]

URLS = list(map(lambda x: URL + x, FILE_NAMES))
SHA256 = [
    '26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316',
    '4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea',
    '2f1f3e594a3066a3084cc57950ca2713c24712adaad03af6ccce18d1846d5618',
]

OUTPUT_FORMAT = 'text:{context_question}\t' 'labels:{answer}'


def _handle_data_point(data_point):
    output = []
    context_question_txt = ""
    for [title, sentences_list] in data_point['context']:
        sentences = '\\n'.join(sentences_list)
        context_question_txt += '{}\\n{}\\n\\n'.format(title, sentences)

    context_question_txt += data_point['question']

    output = OUTPUT_FORMAT.format(
        context_question=context_question_txt, answer=data_point['answer']
    )
    output += '\t\tepisode_done:True\n'
    return output


def make_parlai_format(outpath, dtype, data):
    print('building parlai:' + dtype)
    with open(os.path.join(outpath, dtype + '.txt'), 'w') as fout:
        for data_point in data:
            fout.write(_handle_data_point(data_point))


def build(opt):
    dpath = os.path.join(opt['datapath'], 'HotpotQA')

    if not build_data.built(dpath, version_string=VERSION):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        build_data.download_check(dpath, URLS, FILE_NAMES, SHA256)

        with open(os.path.join(dpath, TRAIN_FILENAME)) as f:
            data = json.load(f)
            make_parlai_format(dpath, 'train', data)

        with open(os.path.join(dpath, DEV_DISTRACTOR_FILENAME)) as f:
            data = json.load(f)
            make_parlai_format(dpath, 'valid_distractor', data)

        with open(os.path.join(dpath, DEV_FULLWIKI_FILENAME)) as f:
            data = json.load(f)
            make_parlai_format(dpath, 'valid_fullwiki', data)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=VERSION)
