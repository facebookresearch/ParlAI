#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.build_data import modelzoo_path
from parlai.utils.io import PathManager


def set_defaults(opt):
    init_model = None
    # check first for 'init_model' for loading model from file
    if opt.get('init_model') and PathManager.exists(opt['init_model']):
        init_model = opt['init_model']
    # next check for 'model_file', this would override init_model
    if opt.get('model_file') and PathManager.exists(opt['model_file']):
        init_model = opt['model_file']

    if init_model is None:
        # Embeddings options
        opt['embedding_file'] = modelzoo_path(
            opt.get('datapath'), opt['embedding_file']
        )
        if opt.get('embedding_file'):
            if not PathManager.exists(opt['embedding_file']):
                raise IOError('No such file: %s' % opt['embedding_file'])
            with PathManager.open(opt['embedding_file']) as f:
                dim = len(f.readline().strip().split(' ')) - 1
                if dim == 1:
                    # first line was a dud
                    dim = len(f.readline().strip().split(' ')) - 1
            opt['embedding_dim'] = dim
        elif not opt.get('embedding_dim'):
            raise RuntimeError(
                ('Either embedding_file or embedding_dim ' 'needs to be specified.')
            )

        # Make sure tune_partial and fix_embeddings are consistent
        if opt['tune_partial'] > 0 and opt['fix_embeddings']:
            print('Setting fix_embeddings to False as tune_partial > 0.')
            opt['fix_embeddings'] = False

        # Make sure fix_embeddings and embedding_file are consistent
        if opt['fix_embeddings'] and not opt.get('embedding_file'):
            print('Setting fix_embeddings to False as embeddings are random.')
            opt['fix_embeddings'] = False


def override_args(opt, override_opt):
    # Major model args are reset to the values in override_opt.
    # Non-architecture args (like dropout) are kept.
    args = set(
        [
            'embedding_file',
            'embedding_dim',
            'hidden_size',
            'doc_layers',
            'question_layers',
            'rnn_type',
            'optimizer',
            'concat_rnn_layers',
            'question_merge',
            'use_qemb',
            'use_in_question',
            'use_tf',
            'vocab_size',
            'num_features',
            'use_time',
        ]
    )
    for k, v in override_opt.items():
        if k in args:
            opt[k] = v
