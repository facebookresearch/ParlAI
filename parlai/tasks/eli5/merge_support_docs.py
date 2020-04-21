# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import sys
from parlai.core.params import ParlaiParser
from data_utils import *



def setup_args():
    parser = ParlaiParser(False, False)
    parser.add_parlai_data_path()
    return parser.parse_args()

def main():
    if ca == 'finalize':
        rd_dir  = pjoin('processed_data/collected_docs', name)
        sl_dir  = pjoin(rd_dir, 'slices')
        if not isdir(sl_dir):
            os.mkdir(sl_dir)
        num_slice   = 0
        docs        = []
        for i in range(10):
            docs    += json.load(open(pjoin(rd_dir, '%d.json' % (i,))))
            while len(docs) > 3000:
                print('writing slice', num_slice, name)
                json.dump(docs[:3000], open(pjoin(sl_dir, 'slice_%d.json' % num_slice), 'w'))
                docs        = docs[3000:]
                num_slice   += 1
        if len(docs) > 0:
            json.dump(docs[:3000], open(pjoin(sl_dir, 'slice_%d.json' % num_slice), 'w'))
    else:
        d_name	= pjoin('processed_data/collected_docs', name, ca)
        if isdir(d_name):
            merged  = merge_support_docs(d_name)
        if len(merged) > 0:
            json.dump(merged, open(pjoin('processed_data/collected_docs', name, ca) + '.json', 'w'))

if __name__ == '__main__':
    print(sys.argv)
    name = sys.argv[1]
    ca = sys.argv[2]
    opt = setup_args()
    print(name, ca, opt['datapath'])
