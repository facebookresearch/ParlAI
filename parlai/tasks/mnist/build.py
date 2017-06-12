# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import gzip
import pickle


data_fname = 'mnist.pkl'

def build(opt):
    dpath = os.path.join(opt['datapath'], 'mnist')

    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = 'mnist_py3k.pkl.gz'
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/' + fname
        build_data.download(url, dpath, fname)
        gunzip(dpath, fname, data_fname)

        # Mark the data as built.
        build_data.mark_done(dpath)

def gunzip(path, fname, outname=None, deleteGzip=True):
    print('unpacking ' + fname)
    fullpath = os.path.join(path, fname)
    if outname is None:
        outname = fname.rsplit('.', 1)[0]
    
    with open(os.path.join(path, outname), 'wb') as f, \
            gzip.open(fullpath) as pkl:
        pickle.dump(pickle.load(pkl), f)

    if deleteGzip:
        os.remove(fullpath)
