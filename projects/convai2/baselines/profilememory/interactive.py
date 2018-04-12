# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Interact with a pre-trained model.
Profile Memory model trained on ConvAI2 using persona 'self'.
"""

from parlai.core.build_data import download_models
from parlai.core.params import ParlaiParser
from parlai.scripts.interactive import interactive
from projects.personachat.persona_seq2seq import PersonachatSeqseqAgentSplit


if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(
        model='projects.personachat.persona_seq2seq:PersonachatSeqseqAgentSplit',
        model_file='models:convai2/profilememory/profilememory_convai2_model',
        dict_file='models:convai2/profilememory/profilememory_convai2.dict',
        interactive_mode=True,
    )

    opt = parser.parse_args()
    opt['model_type'] = 'profilememory' # for builder
    # build profile memory models
    fnames = ['profilememory_convai2_model',
              'profilememory_convai2.dict']
    download_models(opt, fnames, 'convai2', use_model_type=True)
    interactive(opt)
