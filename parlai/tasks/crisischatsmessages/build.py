#!/usr/bin/env python3


# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'crisischatsmessages')
    version = 'None'

    if not build_data.built(dpath, version_string=version):
        
        print('WARNING: NEED TO BUILD DATA -- why is it not already built?')
        
#         print('[building data: ' + dpath + ']')
#         if build_data.built(dpath):
#             # An older version exists, so remove these outdated files.
#             build_data.remove_dir(dpath)
#         build_data.make_dir(dpath)
# 
#         # Download the data.
#         fname = 'empatheticdialogues.tar.gz'
#         url = 'http://parl.ai/downloads/empatheticdialogues/' + fname
#         build_data.download(url, dpath, fname)
#         build_data.untar(dpath, fname)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
