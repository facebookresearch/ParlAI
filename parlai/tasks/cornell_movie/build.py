# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import codecs
import os

def create_fb_format(lines_file, convo_file, outpath):
    print('[building fbformat]')
    with open(os.path.join(outpath, 'train.txt'), 'w') as ftrain, \
            open(os.path.join(outpath, 'valid.txt'), 'w') as fvalid, \
            open(os.path.join(outpath, 'test.txt'), 'w') as ftest:
        lines = {}

        codecs.register_error('strict', codecs.ignore_errors)
        with codecs.open(lines_file, 'r') as f:
            for line in f:
                l = line.split(' +++$+++ ')
                lines[l[0]] = ' '.join(l[4:]).strip('\n').replace('\t', ' ')

        cnt = 0
        with codecs.open(convo_file, 'r') as f:
            for line in f:
                l = line.split(' ')
                convo = ' '.join(l[6:]).strip('\n').strip('[').strip(']')
                c = convo.replace("'",'').replace(' ','').split(',')

                # forward conversation
                s = ''
                index = 0
                for i in range(0, len(c), 2):
                    index += 1
                    s += str(index) + ' ' + lines[c[i]]
                    if len(c) > i + 1:
                        s += '\t' + lines[c[i+1]]
                    s += '\n'

                cnt = cnt + 1
                handle = ftrain
                if (cnt % 10) == 0:
                    handle = ftest
                if (cnt % 10) == 1:
                    handle = fvalid
                handle.write(s + '\n')


def build(opt):
    dpath = os.path.join(opt['datapath'], 'CornellMovie')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = 'cornell_movie_dialogs_corpus.tgz'
        url = 'http://parl.ai/downloads/cornell_movie/' + fname
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)

        dpext = os.path.join(dpath, 'cornell movie-dialogs corpus')
        create_fb_format(os.path.join(dpext, 'movie_lines.txt'),
                         os.path.join(dpext, 'movie_conversations.txt'),
                         dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
