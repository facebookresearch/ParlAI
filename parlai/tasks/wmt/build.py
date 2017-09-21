# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


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
        fnames = [('train.en','train.de', 'en_de_train.txt'),
        ('newstest2014.en','newstest2014.de', 'en_de_test.txt')]
        for (en_fname, de_fname, w_fname) in fnames:
            url_base = 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/'
            en_url = url_base + en_fname
            de_url = url_base + de_fname
            build_data.download(en_url, dpath, en_fname)
            build_data.download(de_url, dpath, de_fname)
            with open(os.path.join(dpath, en_fname), 'r') as f:
                en = [l[:-1] for l in f]

            with open(os.path.join(dpath, de_fname), 'r') as f:
                de = [l[:-1] for l in f]

            with open(os.path.join(dpath, w_fname), 'w') as f:
              for de_sent,en_sent in zip(de,en):
                f.write("1 "+en_sent+"\t"+de_sent+"\n")


        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
