# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Script to run evaluation only on a pretrained model."""
import torch
import logging

from parlai.agents.drqa.utils import Timer
from parlai.agents.drqa.agents import DocReaderAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task

def main(opt):
    # Check options
    opt['datatype'] = 'valid'
    assert('pretrained_model' in opt)

    # Load document reader
    doc_reader = DocReaderAgent(opt)

    # Log params
    logger.info('[ Created with options: ] %s' %
                ''.join(['\n{}\t{}'.format(k, v)
                         for k, v in doc_reader.opt.items()]))

    logger.info('[ Running validation... ]')
    valid_world = create_task(opt, doc_reader)
    valid_time = Timer()
    for _ in valid_world:
        valid_world.parley()

    metrics = valid_world.report()
    logger.info('EM = %.2f | F1 = %.2f | exs = %d' %
                (metrics['accuracy'], metrics['f1'], metrics['total']))
    logger.info('[ Done. Time = %.2f (s) ]' % valid_time.time())


if __name__ == '__main__':
    # Get command line arguments
    argparser = ParlaiParser()
    DocReaderAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()

    # Set logging (only stderr)
    logger = logging.getLogger('DrQA')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Set cuda
    opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
    if opt['cuda']:
        logger.info('[ Using CUDA (GPU %d) ]' % opt['gpu'])
        torch.cuda.set_device(opt['gpu'])

    # Run!
    main(opt)
