# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import gzip
import os


def create_fb_format(inpath, outpath):
    print('[building fbformat]')
    ftrain = open(os.path.join(outpath, 'train.txt'), 'w')
    fvalid = open(os.path.join(outpath, 'valid.txt'), 'w')
    ftest = open(os.path.join(outpath, 'test.txt'), 'w')

    conv_id = 0
    # find all the files.
    for root, _subfolder, files in os.walk(inpath):
        for f in files:
            if f.endswith('.gz'):
                dialog = ''
                conv_id = conv_id + 1
                with gzip.open(os.path.join(root, f), 'r') as f1:
                    # print(str(conv_id) + ': ' + f)
                    words = ''
                    line_id = 1
                    turn_id = 1
                    for line in f1:
                        line = str(line)
                        if line.find('<s id="') != -1:
                            # new sentence
                            if len(words) > 0:
                                if (turn_id % 2) == 0:
                                    dialog += str(line_id) + ' ' + words
                                else:
                                    dialog += '\t' + words + '\n'
                                    line_id += 1
                            turn_id = turn_id + 1
                            words = ''
                        else:
                            i1 = line.find('<w id="')
                            if i1 >= 0:
                                line = line[i1:]
                                word = line[line.find('>')+1:line.find('</w')]
                                words = words + ' ' + word.replace('\t', ' ')
                handle = ftrain
                if (conv_id % 10) == 0:
                    handle = ftest
                if (conv_id % 10) == 1:
                    handle = fvalid
                handle.write(dialog + '\n')

    ftrain.close()
    fvalid.close()
    ftest.close()


def build(opt):
    dpath = os.path.join(opt['datapath'], 'OpenSubtitles')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        url = ('http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/en.tar.gz')
        build_data.download(url, dpath, 'OpenSubtitles.tar.gz')
        build_data.untar(dpath, 'OpenSubtitles.tar.gz')

        create_fb_format(os.path.join(dpath, 'OpenSubtitles', 'en'), dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
