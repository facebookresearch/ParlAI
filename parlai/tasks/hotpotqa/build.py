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
        build_data.download(URL + TRAIN_FILENAME, dpath, TRAIN_FILENAME)
        build_data.download(
            URL + DEV_DISTRACTOR_FILENAME, dpath, DEV_DISTRACTOR_FILENAME
        )
        build_data.download(URL + DEV_FULLWIKI_FILENAME, dpath, DEV_FULLWIKI_FILENAME)

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
