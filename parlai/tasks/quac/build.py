#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import json

VERSION = '0.2'
TRAIN_FILENAME = 'train_v' + VERSION + '.json'
VALID_FILENAME = 'val_v' + VERSION + '.json'

URL = 'https://s3.amazonaws.com/my89public/quac/'
SHOULD = '__SHOULD__'
MAYBE = '__MAYBE__'
SHOULD_NOT = '__SHOULDNOT__'

YES = '__YES__'
NO = '__NO__'
NEITHER = '__NEITHER__'

MAP_CONTINUATION = {'m': MAYBE, 'f': SHOULD, 'n': SHOULD_NOT}
MAP_AFFIRMATION = {'y': YES, 'n': NO, 'x': NEITHER}

OUTPUT_FORMAT = (
    'text:{question}\tfollowup:{continuation}\tyesno:'
    '{affirmation}\tanswer_starts:{start}\tlabels:{labels}'
)


def _parse_answers(q_a):
    starts = []
    labels = []
    for each in q_a['answers']:
        starts.append(str(each['answer_start']))
        labels.append(each['text'].replace('|', ' __PIPE__ '))
    return '|'.join(starts), '|'.join(labels)


def _handle_paragraph(each):
    output = []
    story = each['context'].replace('\n', '\\n')
    for idx, q_a in enumerate(each['qas']):
        question_txt = ''
        if idx == 0:
            question_txt = story + '\\n' + q_a['question']
        else:
            question_txt = q_a['question']
        starts, labels = _parse_answers(q_a)
        output.append(OUTPUT_FORMAT.format(
            question=question_txt,
            continuation=MAP_CONTINUATION.get(q_a['followup']),
            affirmation=MAP_AFFIRMATION.get(q_a['yesno']),
            start=starts,
            labels=labels
        ))
        if idx < len(each['qas']) - 1:
            output.append('\n')
    output.append('\t\tepisode_done:True\n')
    return ''.join(output)


def make_parlai_format(outpath, dtype, data):
    print('building parlai:' + dtype)
    with open(os.path.join(outpath, dtype + '.txt'), 'w') as fout:
        for line in data:
            for each in line['paragraphs']:
                fout.write(_handle_paragraph(each))


def build(opt):
    dpath = os.path.join(opt['datapath'], 'QuAC')
    version = VERSION

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        build_data.download(URL + TRAIN_FILENAME, dpath, TRAIN_FILENAME)
        build_data.download(URL + VALID_FILENAME, dpath, VALID_FILENAME)

        with open(os.path.join(dpath, TRAIN_FILENAME)) as f:
            data = json.load(f)['data']
            make_parlai_format(dpath, 'train', data)

        with open(os.path.join(dpath, VALID_FILENAME)) as f:
            data = json.load(f)['data']
            make_parlai_format(dpath, 'valid', data)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
