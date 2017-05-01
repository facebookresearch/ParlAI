# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data


def _process(fname, fout):
    with open(fname) as f:
        lines = [line.strip('\n') for line in f]
    # main article
    s = '1 ' + lines[2]
    # add question
    s = s + lines[4]
    # add answer
    s = s + '\t' + lines[6]
    # add candidates (and strip them of the real names)
    for i in range(8, len(lines)):
        lines[i] = lines[i].split(':')[0]
    s = s + '\t\t' + '|'.join(lines[8:-1])
    fout.write(s + '\n\n')


def create_fb_format(outpath, dtype, inpath):
    print('building fbformat:' + dtype)
    import os
    fout = open(outpath + dtype + '.txt', 'w')
    for file in os.listdir(inpath):
        if file.endswith('.question'):
            fname = os.path.join(inpath, file)
            _process(fname, fout)
    fout.close()


def build(opt):
    dpath = opt['datapath'] + '/QACNN/'

    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = 'cnn.tgz'
        gd_id = '0BwmD_VLjROrfTTljRDVZMFJnVWM'
        build_data.download_file_from_google_drive(gd_id, dpath + fname)
        build_data.untar(dpath, fname)

        create_fb_format(dpath, 'train', dpath + 'cnn/questions/training/')
        create_fb_format(dpath, 'valid', dpath + 'cnn/questions/validation/')
        create_fb_format(dpath, 'test', dpath + 'cnn/questions/test/')

        # Mark the data as built.
        build_data.mark_done(dpath)
