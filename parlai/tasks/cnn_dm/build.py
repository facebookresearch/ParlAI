#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import hashlib

CNN_ROOT = 'https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/'
DM_ROOT = 'https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/'

CNN_FNAMES = {
    'train': 'cnn_wayback_training_urls.txt',
    'valid': 'cnn_wayback_validation_urls.txt',
    'test': 'cnn_wayback_test_urls.txt',
}
DM_FNAMES = {
    'train': 'dailymail_wayback_training_urls.txt',
    'valid': 'dailymail_wayback_validation_urls.txt',
    'test': 'dailymail_wayback_test_urls.txt',
}


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

        cnn_fname = 'cnn_stories.tgz'
        cnn_gd_id = '0BwmD_VLjROrfTHk4NFg2SndKcjQ'
        build_data.download_from_google_drive(cnn_gd_id, os.path.join(dpath, cnn_fname))
        build_data.untar(dpath, cnn_fname)

        dm_fname = 'dm_stories.tgz'
        dm_gd_id = '0BwmD_VLjROrfM1BxdkxVaTY2bWs'
        build_data.download_from_google_drive(dm_gd_id, os.path.join(dpath, dm_fname))
        build_data.untar(dpath, dm_fname)

        for dt in CNN_FNAMES:
            fname = CNN_FNAMES[dt]
            url = CNN_ROOT + fname
            build_data.download(url, dpath, fname)
            urls_fname = os.path.join(dpath, fname)
            split_fname = os.path.join(dpath, dt + '.txt')
            with open(urls_fname) as urls_file, open(split_fname, 'a') as split_file:
                for url in urls_file:
                    file_name = hashlib.sha1(url.strip().encode('utf-8')).hexdigest()
                    split_file.write("cnn/stories/{}.story\n".format(file_name))

        for dt in DM_FNAMES:
            fname = DM_FNAMES[dt]
            url = DM_ROOT + fname
            build_data.download(url, dpath, fname)
            urls_fname = os.path.join(dpath, fname)
            split_fname = os.path.join(dpath, dt + '.txt')
            with open(urls_fname) as urls_file, open(split_fname, 'a') as split_file:
                for url in urls_file:
                    file_name = hashlib.sha1(url.strip().encode('utf-8')).hexdigest()
                    split_file.write("dailymail/stories/{}.story\n".format(file_name))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
