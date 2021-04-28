#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pre-trained DPR Model.
"""
import gzip
import os
import os.path

from parlai.core.build_data import built, download as download_path, mark_done
import parlai.utils.logging as logger

path = 'https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz'


def download(datapath):
    dpath = os.path.join(datapath, 'models/hallucination/wiki_passages')
    fname = 'psgs_w100.tsv.gz'
    gzip_file = os.path.join(dpath, fname)
    new_file = os.path.join(dpath, fname.replace('.gz', ''))
    version = 'v1.0'
    if not built(dpath, version):
        os.makedirs(dpath)
        download_path(path, dpath, fname)
        input = gzip.GzipFile(gzip_file, "rb")
        s = input.read()
        input.close()
        output = open(new_file, "wb")
        output.write(s)
        output.close()
        logger.info(f" Saved to {new_file}")
        mark_done(dpath, version)
