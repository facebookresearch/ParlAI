#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import gzip
import os
import re


def _regularize(sent):
    sent = sent.replace('i&gt;', '').replace('&lt;', '').replace('&gt;', '')
    sent = re.sub(r'x[0-9|a-f][0-9|a-f]', ' ', sent)
    sent = sent.replace('\\', '').replace('-', '')
    sent = ' '.join(re.findall(r"[\w']+|[.,!?:;]", sent))
    sent = sent.replace('. .', '...')
    sent = ' '.join(sent.split())
    return sent


def create_fb_format(inpath, outpath):
    print('[building fbformat]')
    with open(os.path.join(outpath, 'train.txt'), 'w') as ftrain, \
            open(os.path.join(outpath, 'valid.txt'), 'w') as fvalid, \
            open(os.path.join(outpath, 'test.txt'), 'w') as ftest:

        conv_id = 0
        # find all the files.
        for root, _subfolder, files in os.walk(inpath):
            for f in files:
                if f.endswith('.gz'):
                    dialog = []
                    conv_id = conv_id + 1
                    with gzip.open(os.path.join(root, f), 'r') as f1:
                        words = []
                        line_id = 1
                        turn_id = 0
                        for line in f1:
                            line = str(line)
                            if line.find('<s id="') != -1:
                                # new sentence
                                if len(words) > 0:
                                    curr_words = _regularize(''.join(words))
                                    if len(curr_words) > 0:
                                        if (turn_id % 2) == 0:
                                            dialog.append(str(line_id))
                                            dialog.append(' ')
                                            dialog.append(curr_words)
                                        else:
                                            dialog.append('\t')
                                            dialog.append(curr_words)
                                            dialog.append('\n')
                                            line_id += 1
                                        turn_id += + 1
                                words.clear()
                            else:
                                i1 = line.find('<w id="')
                                if i1 >= 0:
                                    line = line[i1:]
                                    word = line[line.find('>') + 1:line.find('</w')]
                                    words.append(' ')
                                    words.append(word.replace('\t', ' '))
                    handle = ftrain
                    if (conv_id % 10) == 0:
                        handle = ftest
                    if (conv_id % 10) == 1:
                        handle = fvalid
                    dialog.append('\n')
                    handle.write(''.join(dialog))


def build(datapath):
    dpath = os.path.join(datapath, 'OpenSubtitles')
    version = '2'

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
    return dpath
