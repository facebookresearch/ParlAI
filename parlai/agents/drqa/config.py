# Copyright 2004-present Facebook. All Rights Reserved.
import os
import sys
import pathlib
import logging
logger = logging.getLogger('DrQA')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_cmdline_args(parser):
    # Parlai root directory
    parlai_dir = pathlib.Path(__file__).parents[3].as_posix()

    # Runtime environment
    parser.add_argument('--no_cuda', type='bool', default=False)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--random_seed', type=int, default=1013)

    # Basics
    parser.add_argument('--model_file', type=str, default=None,
                        help='Path where best valid models are saved')
    parser.add_argument('--embedding_file', type=str,
                        default=parlai_dir + '/data/GloVe/glove.840B.300d.txt',
                        help='File of space separated embeddings: w e1 ... ed')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Load dict/features/weights/opts from this file')
    parser.add_argument('--log_file', type=str, default=None)

    # Model details
    parser.add_argument('--fix_embeddings', type='bool', default=True)
    parser.add_argument('--tune_partial', type=int, default=0,
                        help='Train the K most frequent word embeddings')
    parser.add_argument('--embedding_dim', type=int, default=None,
                        help=('Default embedding size if '
                              'embedding_file is not given'))
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size of RNN units')
    parser.add_argument('--doc_layers', type=int, default=3,
                        help='Number of RNN layers for passage')
    parser.add_argument('--question_layers', type=int, default=3,
                        help='Number of RNN layers for question')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        help='RNN type: lstm (default), gru, or rnn')

    # Optimization details
    parser.add_argument('--valid_metric', type=str,
                        choices=['accuracy', 'f1'], default='f1',
                        help='Metric for choosing best valid model')
    parser.add_argument('--max_len', type=int, default=15,
                        help='The max span allowed during decoding')
    parser.add_argument('--rnn_padding', type='bool', default=False)
    parser.add_argument('--display_iter', type=int, default=10,
                        help='Print train error after every \
                              <display_iter> epoches (default 10)')
    parser.add_argument('--dropout_emb', type=float, default=0.3,
                        help='Dropout rate for word embeddings')
    parser.add_argument('--dropout_rnn', type=float, default=0.3,
                        help='Dropout rate for RNN states')
    parser.add_argument('--dropout_rnn_output', type='bool', default=True,
                        help='Whether to dropout the RNN output')
    parser.add_argument('--optimizer', type=str, default='adamax',
                        help='Optimizer: sgd or adamax (default)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1,
                        help='Learning rate for SGD (default 0.1)')
    parser.add_argument('--grad_clipping', type=float, default=10,
                        help='Gradient clipping (default 10.0)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (default 0)')
    parser.add_argument('--momentum', type=float, default=0,
                        help='Momentum (default 0)')

    # Model-specific
    parser.add_argument('--concat_rnn_layers', type='bool', default=True)
    parser.add_argument('--question_merge', type=str, default='self_attn',
                        help='The way of computing question representation')
    parser.add_argument('--use_qemb', type='bool', default=True,
                        help='Whether to use weighted question embeddings')
    parser.add_argument('--use_in_question', type='bool', default=True,
                        help='Whether to use in_question features')
    parser.add_argument('--use_tf', type='bool', default=True,
                        help='Whether to use tf features')


def set_defaults(opt):
    # Check critical files exist
    if not os.path.isfile(opt['embedding_file']):
        raise IOError('No such file: %s' % args.embedding_file)

    # Embeddings options
    if 'embedding_file' in opt:
        with open(opt['embedding_file']) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        if 'embedding_dim' in opt and opt['embedding_dim'] != dim:
            raise ValueError('embedding_dim = %d, but %s has %d dims.' %
                             (opt['embedding_dim'], opt['embedding_file'], dim))
        opt['embedding_dim'] = dim
    elif 'embedding_dim' not in opt:
        raise RuntimeError(('Either embedding_file or embedding_dim '
                            'needs to be specified.'))

    # Make sure tune_partial and fix_embeddings are consistent
    if opt['tune_partial'] > 0 and opt['fix_embeddings']:
        logger.info('Setting fix_embeddings to False as tune_partial > 0.')
        opt['fix_embeddings'] = False


def override_args(opt, override_opt):
    # Major model args are reset to the values in override_opt.
    # Non-architecture args (like dropout) are kept.
    args = set(['embedding_file', 'embedding_dim', 'hidden_size', 'doc_layers',
                'question_layers', 'rnn_type', 'optimizer', 'concat_rnn_layers',
                'question_merge', 'use_qemb', 'use_in_question', 'use_tf'])
    for k, v in override_opt.items():
        if k in args:
            opt[k] = v
