#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import numpy
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en',
        'train.en',
        '845ee390042259f7512eabc6458b0fdb30db28d254c83232d97d4161c1fdae51',
        zipped=False,
    ),
    DownloadableFile(
        'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de',
        'train.de',
        'a2e292ad1b1f3fec6224dc043460ba6c453932f470109579b8c1ce6d4df65262',
        zipped=False,
    ),
    DownloadableFile(
        'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en',
        'newstest2014.en',
        '2db4575449877142aef9187e5e8f58ec10af73a2589ad7a4690208f5234901bb',
        zipped=False,
    ),
    DownloadableFile(
        'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de',
        'newstest2014.de',
        '39712f5709343ab17e8daf341cb99d58bf8c0e626e29bbae6d7dffd481114f8b',
        zipped=False,
    ),
]


def readFiles(dpath, rfnames):
    en_fname, de_fname = rfnames
    with open(os.path.join(dpath, en_fname), 'r') as f:
        # We replace '##AT##-##AT##' as a workaround in order to use the
        # nltk tokenizer specified by DictionaryAgent
        en = [l[:-1].replace('##AT##-##AT##', '__AT__') for l in f]

    with open(os.path.join(dpath, de_fname), 'r') as f:
        de = [l[:-1].replace('##AT##-##AT##', '__AT__') for l in f]

    return list(zip(de, en))


def build(opt):
    dpath = os.path.join(opt['datapath'], 'wmt')
    version = 'None'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        train_r_fnames = ('train.en', 'train.de')
        train_w_fname = 'en_de_train.txt'
        valid_w_fname = 'en_de_valid.txt'
        test_r_fnames = ('newstest2014.en', 'newstest2014.de')
        test_w_fname = 'en_de_test.txt'

        train_zip = readFiles(dpath, train_r_fnames)
        numpy.random.shuffle(train_zip)
        with open(os.path.join(dpath, valid_w_fname), 'w') as f:
            for de_sent, en_sent in train_zip[:30000]:
                f.write('1 ' + en_sent + '\t' + de_sent + '\n')
        with open(os.path.join(dpath, train_w_fname), 'w') as f:
            for de_sent, en_sent in train_zip[30000:]:
                f.write('1 ' + en_sent + '\t' + de_sent + '\n')

        test_zip = readFiles(dpath, test_r_fnames)
        with open(os.path.join(dpath, test_w_fname), 'w') as f:
            for de_sent, en_sent in test_zip:
                f.write('1 ' + en_sent + '\t' + de_sent + '\n')

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
