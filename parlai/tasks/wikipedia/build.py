# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import parlai.core.build_data as build_data
import os
from .wikiextractor import main as wikiextract


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'wikipedia')

    task = opt.get('task', 'wikipedia:all')
    extract_full = task.split(':')[-1] == 'all'
    if extract_full:
        dpath = os.path.join(dpath, 'full')
    else:
        dpath = os.path.join(dpath, 'summary')
    # check if data had been previously built
    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')

        build_data.make_dir(dpath)

        if extract_full:
            # download the data.
            fname = 'enwiki-latest-pages-articles.xml.bz2'
            url = 'https://dumps.wikimedia.org/enwiki/latest/' + fname
            build_data.download(url, dpath, fname)

            # uncompress it


            # args = {}
            # args['input'] = dpath + '/' + fname
            # args['filter_disambig_pages'] = True
            # args['processes'] = max(1, os.cpu_count() - 1)
            # args['output'] = dpath + '/' + 'wiki_extracted'
            # args['json'] = True
            #
            # wikiextract(args)
            # # mark the data as built
            build_data.mark_done(dpath)
            return False
        else:
            fname = "TBD for Our stuff"
            url = "url_tbd" + fname
            build_data.download(url, dpath, fname)
            build_data.mark_done(dpath)
            return True
    return True
