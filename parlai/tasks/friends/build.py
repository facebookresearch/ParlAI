#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
from parlai.core.build_data import DownloadableFile
import os
import jsonlines
from collections import defaultdict
from sklearn.model_selection import train_test_split

RANDOM_SEED = 123

RESOURCES = [
    DownloadableFile(
        'http://zissou.infosci.cornell.edu/convokit/datasets/friends-corpus/friends-corpus.zip',
        'friends-corpus.zip',
        '51ae80ce345212839d256b59b4982e9b40229ff6049115bd54d885a285d2b921',
        zipped=True,
    )
]


def generate_folds(dpath):
    """
    Generate Data Folds based on the scene id.
    """
    datafile = os.path.join(dpath, 'friends-corpus/utterances.jsonl')
    train_datafile = os.path.join(dpath, 'train.jsonl')
    valid_datafile = os.path.join(dpath, 'valid.jsonl')
    test_datafile = os.path.join(dpath, 'test.jsonl')

    # Load the dataset
    conversations = defaultdict(list)
    with jsonlines.open(datafile) as reader:
        for utterance in reader:
            text = utterance['text']
            speaker = utterance['speaker']
            conversation_id = utterance['conversation_id']

            if speaker != 'TRANSCRIPT_NOTE':
                conversations[conversation_id].append(
                    {
                        "text": text,
                        "speaker": speaker,
                        "conversation_id": conversation_id,
                    }
                )

    # Split the dataset into 80% train, 10% valid, 10% test
    train, valid_and_test = train_test_split(
        list(conversations.keys()), test_size=0.2, random_state=RANDOM_SEED
    )
    valid, test = train_test_split(
        valid_and_test, test_size=0.5, random_state=RANDOM_SEED
    )

    # Save the data folds into separate files
    with jsonlines.open(train_datafile, mode='w') as writer:
        for conversation_id in train:
            for utterance in conversations[conversation_id]:
                writer.write(utterance)

    with jsonlines.open(valid_datafile, mode='w') as writer:
        for conversation_id in valid:
            for utterance in conversations[conversation_id]:
                writer.write(utterance)

    with jsonlines.open(test_datafile, mode='w') as writer:
        for conversation_id in test:
            for utterance in conversations[conversation_id]:
                writer.write(utterance)


def build(opt):
    dpath = os.path.join(opt['datapath'], 'Friends')
    version = '1.00'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        generate_folds(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
