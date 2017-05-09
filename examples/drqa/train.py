# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Trains (a partial) implementation of the DrQa Document Reader from:

Danqi Chen, Adam Fisch, Jason Weston, Antoine Bordes. 2017.
Reading Wikipedia to Answer Open-Domain Questions.
In Association for Computational Linguistics (ACL).

Link: https://arxiv.org/abs/1704.00051

Note:
To use pretrained word embeddings, set the --embeddings_file path argument.
GloVe is recommended, see http://nlp.stanford.edu/data/glove.840B.300d.zip.
"""
try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('Need to install pytorch: go to pytorch.org')
import numpy as np
import logging
import copy
import sys

from parlai.agents.drqa.agents import SimpleDictionaryAgent
from parlai.agents.drqa.agents import DocReaderAgent
from parlai.agents.drqa.utils import Timer
from parlai.core.worlds import DialogPartnerWorld
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task


def build_dict(opt):
    opt = copy.deepcopy(opt)
    opt['batchsize'] = 1
    dictionary = SimpleDictionaryAgent(opt)

    # We use the train set to build the dictionary.
    logger.info('[ Building dictionary... ]')
    opt['datatype'] = 'train:ordered'
    world = create_task(opt, dictionary)
    for _ in world:
        world.parley()

    dictionary.sort()
    logger.info('[ Dictionary built. ]')
    logger.info('[ Num words = %d ]' % len(dictionary))

    return dictionary


def validate(opt, agent, n_iter):
    opt = copy.deepcopy(opt)
    opt['datatype'] = 'valid'
    opt['batchsize'] = 1
    valid_world = create_task(opt, agent)

    logger.info('[ Running validation... ]')
    valid_time = Timer()
    for _ in valid_world:
        valid_world.parley()

    metrics = valid_world.report()
    logger.info('[valid] iter = %d | EM = %.2f | F1 = %.2f | exs = %d' %
                (n_iter, metrics['accuracy'], metrics['f1'], metrics['total']))
    logger.info('[ Done. Time = %.2f (s) ]' % valid_time.time())

    return metrics[opt['valid_metric']]


def main(opt):
    # Build dictionary from task data
    if 'pretrained_model' in opt:
        dictionary = None
    else:
        dictionary = build_dict(opt)

    # Build document reader
    doc_reader = DocReaderAgent(opt, word_dict=dictionary)

    # Log params
    logger.info('[ Created with options: ] %s' %
                ''.join(['\n{}\t{}'.format(k, v)
                         for k, v in doc_reader.items()]))

    # Build training world once
    opt['datatype'] = 'train'
    train_world = create_task(opt, doc_reader)
    train_time = Timer()

    # Keep track of best model + how long since the last improvement
    best_valid = 0
    impatience = 0

    logger.info("[ Ok, let's go... ]")
    iteration = 0
    while impatience < opt['patience']:
        # Train...
        logger.info('[ Training for %d iters... ]' % opt['train_interval'])
        train_time.reset()
        for _ in range(opt['train_interval']):
            train_world.parley()
        logger.info('[ Done. Time = %.2f (s) ]' % train_time.time())

        # ...validate!
        valid_metric = validate(opt, doc_reader, iteration)
        if valid_metric > best_valid:
            logger.info(
                '[ Best eval %d: %s = %.2f (old = %.2f) ]' %
                (iteration, opt['valid_metric'], valid_metric, best_valid)
            )
            best_valid = valid_metric
            impatience = 0
            if 'model_file' in opt:
                doc_reader.save(opt['model_file'])
        else:
            impatience += 1

        iteration += 1


if __name__ == '__main__':
    # Get command line arguments
    argparser = ParlaiParser()
    argparser.add_arg(
        '--train_interval', type=int, default=1000,
        help='Validate after every N train updates',
    )
    argparser.add_arg(
        '--patience', type=int, default=10,
        help='Number of intervals to continue without improvement'
    )
    SimpleDictionaryAgent.add_cmdline_args(argparser)
    DocReaderAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()

    # Set logging
    logger = logging.getLogger('DrQA')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if 'log_file' in opt:
        logfile = logging.FileHandler(opt['log_file'], 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('[ COMMAND: %s ]' % ' '.join(sys.argv))

    # Set cuda
    opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
    if opt['cuda']:
        logger.info('[ Using CUDA (GPU %d) ]' % opt['gpu'])
        torch.cuda.set_device(opt['gpu'])

    # Set random state
    np.random.seed(opt['random_seed'])
    torch.manual_seed(opt['random_seed'])
    if opt['cuda']:
        torch.cuda.manual_seed(opt['random_seed'])

    # Run!
    main(opt)
