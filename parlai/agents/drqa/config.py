#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from parlai.core.build_data import modelzoo_path


def add_cmdline_args(parser):
    # Runtime environment
    agent = parser.add_argument_group('DrQA Arguments')
    agent.add_argument('--no_cuda', type='bool', default=False)
    agent.add_argument('--gpu', type=int, default=-1)
    agent.add_argument('--random_seed', type=int, default=1013)

    # Basics
    agent.add_argument('--embedding_file', type=str, default=None,
                       help='file of space separated embeddings: w e1 ... ed')
    agent.add_argument('--init_model', type=str, default=None,
                       help='Load dict/features/weights/opts from this file')
    agent.add_argument('--log_file', type=str, default=None)

    # Model details
    agent.add_argument('--fix_embeddings', type='bool', default=True)
    agent.add_argument('--tune_partial', type=int, default=0,
                       help='Train the K most frequent word embeddings')
    agent.add_argument('--embedding_dim', type=int, default=300,
                       help=('Default embedding size if '
                             'embedding_file is not given'))
    agent.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden size of RNN units')
    agent.add_argument('--doc_layers', type=int, default=3,
                       help='Number of RNN layers for passage')
    agent.add_argument('--question_layers', type=int, default=3,
                       help='Number of RNN layers for question')
    agent.add_argument('--rnn_type', type=str, default='lstm',
                       help='RNN type: lstm (default), gru, or rnn')

    # Optimization details
    agent.add_argument('--valid_metric', type=str,
                       choices=['accuracy', 'f1'], default='f1',
                       help='Metric for choosing best valid model')
    agent.add_argument('--max_len', type=int, default=15,
                       help='The max span allowed during decoding')
    agent.add_argument('--rnn_padding', type='bool', default=False)
    agent.add_argument('--display_iter', type=int, default=10,
                       help='Print train error after every'
                            '<display_iter> epoches (default 10)')
    agent.add_argument('--dropout_emb', type=float, default=0.4,
                       help='Dropout rate for word embeddings')
    agent.add_argument('--dropout_rnn', type=float, default=0.4,
                       help='Dropout rate for RNN states')
    agent.add_argument('--dropout_rnn_output', type='bool', default=True,
                       help='Whether to dropout the RNN output')
    agent.add_argument('--optimizer', type=str, default='adamax',
                       help='Optimizer: sgd or adamax (default)')
    agent.add_argument('--learning_rate', '-lr', type=float, default=0.1,
                       help='Learning rate for SGD (default 0.1)')
    agent.add_argument('--grad_clipping', type=float, default=10,
                       help='Gradient clipping (default 10.0)')
    agent.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay (default 0)')
    agent.add_argument('--momentum', type=float, default=0,
                       help='Momentum (default 0)')
    agent.add_argument('--subsample-docs', type=int, default=0,
                       help='When given paragraphs (separated by \n) will take '
                            'only a subset of them, including the one with the '
                            'answer for training.')

    # Model-specific
    agent.add_argument('--concat_rnn_layers', type='bool', default=True)
    agent.add_argument('--question_merge', type=str, default='self_attn',
                       help='The way of computing question representation')
    agent.add_argument('--use_qemb', type='bool', default=True,
                       help='Whether to use weighted question embeddings')
    agent.add_argument('--use_in_question', type='bool', default=True,
                       help='Whether to use in_question features')
    agent.add_argument('--use_tf', type='bool', default=True,
                       help='Whether to use tf features')
    agent.add_argument('--use_time', type=int, default=0,
                       help='Time features marking how recent word was said')


def set_defaults(opt):
    init_model = None
    # check first for 'init_model' for loading model from file
    if opt.get('init_model') and os.path.isfile(opt['init_model']):
        init_model = opt['init_model']
    # next check for 'model_file', this would override init_model
    if opt.get('model_file') and os.path.isfile(opt['model_file']):
        init_model = opt['model_file']

    if init_model is None:
        # Embeddings options
        opt['embedding_file'] = modelzoo_path(
            opt.get('datapath'), opt['embedding_file'])
        if opt.get('embedding_file'):
            if not os.path.isfile(opt['embedding_file']):
                raise IOError('No such file: %s' % opt['embedding_file'])
            with open(opt['embedding_file']) as f:
                dim = len(f.readline().strip().split(' ')) - 1
                if dim == 1:
                    # first line was a dud
                    dim = len(f.readline().strip().split(' ')) - 1
            opt['embedding_dim'] = dim
        elif not opt.get('embedding_dim'):
            raise RuntimeError(('Either embedding_file or embedding_dim '
                                'needs to be specified.'))

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
    args = set(['embedding_file', 'embedding_dim', 'hidden_size', 'doc_layers',
                'question_layers', 'rnn_type', 'optimizer', 'concat_rnn_layers',
                'question_merge', 'use_qemb', 'use_in_question', 'use_tf',
                'vocab_size', 'num_features', 'use_time'])
    for k, v in override_opt.items():
        if k in args:
            opt[k] = v
