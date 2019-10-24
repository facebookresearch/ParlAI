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

URLS = ['0BwmD_VLjROrfTHk4NFg2SndKcjQ', '0BwmD_VLjROrfM1BxdkxVaTY2bWs']

FILE_NAMES = [
    'cnn_stories.tgz',
    'dm_stories.tgz',
    'cnn_wayback_training_urls.txt',
    'cnn_wayback_validation_urls.txt',
    'cnn_wayback_test_urls.txt',
    'dailymail_wayback_training_urls.txt',
    'dailymail_wayback_validation_urls.txt',
    'dailymail_wayback_test_urls.txt'
]

URLS += list(map(lambda x: CNN_ROOT + x, FILE_NAMES[2:5]))
URLS += list(map(lambda x: DM_ROOT + x, FILE_NAMES[5:]))

SHA256 = [
    'e8fbc0027e54e0a916abd9c969eb35f708ed1467d7ef4e3b17a56739d65cb200',
    'ad69010002210b7c406718248ee66e65868b9f6820f163aa966369878d14147e',
    'e074c2245c475b00c455cefb911e0066b27fe17085dd0c773101e10d3088583b',
    'b1ae81ff058ca640da3ae2b3c98fefca3adfea358736b6e29efc2ec1cbef5b5c',
    'a0796c3c7812e3c9fcb1a65faa9aee7bb6f8a3869e953c7f61b401790c0a6f33',
    '3913d6a90c29a81196128346d81c28d6c7f7e91777d886e8417163ce83b2a04a',
    '2377b8f809bd07b143bbbd9e60594d10e7b8a211c8a5672181ea6000bbf548a2',
    '554d18fc79a06a16902662d926cb7cc981ea36a3f82d5ae1426e25bf62f65b87'
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
        build_data.download_check(dpath, URLS[:2], FILE_NAMES[:2], SHA256[:2], from_google=True)
        build_data.download_check(dpath, URLS[2:], FILE_NAMES[2:], SHA256[2:])

        for zipfile in FILE_NAMES[:2]:
            build_data.untar(dpath, zipfile)

        for i, (fname, url) in enumerate(zip(FILE_NAMES[2:5], URLS[2:5])):
            dt = data_type[i]
            url = CNN_ROOT + fname
            build_data.download(url, dpath, fname)
            urls_fname = os.path.join(dpath, fname)
            split_fname = os.path.join(dpath, dt + '.txt')
            with open(urls_fname) as urls_file, open(split_fname, 'a') as split_file:
                for url in urls_file:
                    file_name = hashlib.sha1(url.strip().encode('utf-8')).hexdigest()
                    split_file.write("cnn/stories/{}.story\n".format(file_name))

        for i, (fname, url) in enumerate(zip(FILE_NAMES[5:], URLS[5:])):
            dt = data_type[i]
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

# def build(opt):
#     dpath = os.path.join(opt['datapath'], 'CNN_DM')
#     version = None

#     if not build_data.built(dpath, version_string=version):
#         print('[building data: ' + dpath + ']')
#         if build_data.built(dpath):
#             # An older version exists, so remove these outdated files.
#             build_data.remove_dir(dpath)
#         build_data.make_dir(dpath)

#         # Download the data.

#         cnn_fname = 'cnn_stories.tgz'
#         cnn_gd_id = '0BwmD_VLjROrfTHk4NFg2SndKcjQ'
#         build_data.download_from_google_drive(cnn_gd_id, os.path.join(dpath, cnn_fname))
#         # build_data.untar(dpath, cnn_fname)

#         dm_fname = 'dm_stories.tgz'
#         dm_gd_id = '0BwmD_VLjROrfM1BxdkxVaTY2bWs'
#         build_data.download_from_google_drive(dm_gd_id, os.path.join(dpath, dm_fname))
#         # build_data.untar(dpath, dm_fname)

#         for dt in CNN_FNAMES:
#             fname = CNN_FNAMES[dt]
#             url = CNN_ROOT + fname
#             build_data.download(url, dpath, fname)
#             urls_fname = os.path.join(dpath, fname)
#             split_fname = os.path.join(dpath, dt + '.txt')
#             with open(urls_fname) as urls_file, open(split_fname, 'a') as split_file:
#                 for url in urls_file:
#                     file_name = hashlib.sha1(url.strip().encode('utf-8')).hexdigest()
#                     split_file.write("cnn/stories/{}.story\n".format(file_name))

#         for dt in DM_FNAMES:
#             fname = DM_FNAMES[dt]
#             url = DM_ROOT + fname
#             build_data.download(url, dpath, fname)
#             urls_fname = os.path.join(dpath, fname)
#             split_fname = os.path.join(dpath, dt + '.txt')
#             with open(urls_fname) as urls_file, open(split_fname, 'a') as split_file:
#                 for url in urls_file:
#                     file_name = hashlib.sha1(url.strip().encode('utf-8')).hexdigest()
#                     split_file.write("dailymail/stories/{}.story\n".format(file_name))

#         # Mark the data as built.
#         build_data.mark_done(dpath, version_string=version)
