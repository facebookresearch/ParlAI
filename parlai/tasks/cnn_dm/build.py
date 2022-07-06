#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import hashlib
from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager

RESOURCES = [
    DownloadableFile(
        '0BwmD_VLjROrfTHk4NFg2SndKcjQ',
        'cnn_stories.tgz',
        'e8fbc0027e54e0a916abd9c969eb35f708ed1467d7ef4e3b17a56739d65cb200',
        from_google=True,
    ),
    DownloadableFile(
        '0BwmD_VLjROrfM1BxdkxVaTY2bWs',
        'dm_stories.tgz',
        'ad69010002210b7c406718248ee66e65868b9f6820f163aa966369878d14147e',
        from_google=True,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/cnn_wayback_training_urls.txt',
        'cnn_wayback_training_urls.txt',
        'e074c2245c475b00c455cefb911e0066b27fe17085dd0c773101e10d3088583b',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/cnn_wayback_validation_urls.txt',
        'cnn_wayback_validation_urls.txt',
        'b1ae81ff058ca640da3ae2b3c98fefca3adfea358736b6e29efc2ec1cbef5b5c',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/cnn_wayback_test_urls.txt',
        'cnn_wayback_test_urls.txt',
        'a0796c3c7812e3c9fcb1a65faa9aee7bb6f8a3869e953c7f61b401790c0a6f33',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/dailymail_wayback_training_urls.txt',
        'dailymail_wayback_training_urls.txt',
        '3913d6a90c29a81196128346d81c28d6c7f7e91777d886e8417163ce83b2a04a',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/dailymail_wayback_validation_urls.txt',
        'dailymail_wayback_validation_urls.txt',
        '2377b8f809bd07b143bbbd9e60594d10e7b8a211c8a5672181ea6000bbf548a2',
        zipped=False,
    ),
    DownloadableFile(
        'https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/dailymail_wayback_test_urls.txt',
        'dailymail_wayback_test_urls.txt',
        '554d18fc79a06a16902662d926cb7cc981ea36a3f82d5ae1426e25bf62f65b87',
        zipped=False,
    ),
]

data_type = ['train', 'valid', 'test']


def build(opt):
    dpath = os.path.join(opt['datapath'], 'CNN_DM')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        for i, f in enumerate(RESOURCES[2:5]):
            dt = data_type[i]
            urls_fname = os.path.join(dpath, f.file_name)
            split_fname = os.path.join(dpath, dt + '.txt')
            with PathManager.open(urls_fname) as urls_file, PathManager.open(
                split_fname, 'a'
            ) as split_file:
                for url in urls_file:
                    file_name = hashlib.sha1(url.strip().encode('utf-8')).hexdigest()
                    split_file.write("cnn/stories/{}.story\n".format(file_name))

        for i, f in enumerate(RESOURCES[5:]):
            dt = data_type[i]
            urls_fname = os.path.join(dpath, f.file_name)
            split_fname = os.path.join(dpath, dt + '.txt')
            with PathManager.open(urls_fname) as urls_file, PathManager.open(
                split_fname, 'a'
            ) as split_file:
                for url in urls_file:
                    file_name = hashlib.sha1(url.strip().encode('utf-8')).hexdigest()
                    split_file.write("dailymail/stories/{}.story\n".format(file_name))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
