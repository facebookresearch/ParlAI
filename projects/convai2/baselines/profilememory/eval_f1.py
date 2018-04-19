# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained profile memory model trained on convai2:self."""

from parlai.core.build_data import download_models
from projects.convai2.eval_f1 import setup_args, eval_f1


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        dict_file='models:convai2/profilememory/profilememory_convai2.dict',
        model='projects.personachat.persona_seq2seq:PersonachatSeqseqAgentSplit',
        model_file='models:convai2/profilememory/profilememory_convai2_model',
        rank_candidates=False,
    )

    opt = parser.parse_args(print_args=False)
    opt['model_type'] = 'profilememory'
    # build profile memory models
    fnames = ['profilememory_convai2_model',
              'profilememory_convai2_ppl_model',
              'profilememory_convai2.dict']
    download_models(opt, fnames, 'convai2', version='v2.0', use_model_type=True)

    eval_f1(opt, print_parser=parser)
