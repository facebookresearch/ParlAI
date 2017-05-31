# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Simple interactive script to play with a pretrained model. User provides
context and question pairs, agent supplies an answer.

Example interaction:
Context: I was thirsty today. So I went to the market and bought some water.
Question: What did I buy?
Reply: some water
"""
try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('Need to install pytorch: go to pytorch.org')
import logging

from parlai.agents.drqa.agents import DocReaderAgent
from parlai.core.params import ParlaiParser

def main(opt):
    # Load document reader
    assert('pretrained_model' in opt)
    doc_reader = DocReaderAgent(opt)

    # Log params
    logger.info('[ Created with options: ] %s' %
                ''.join(['\n{}\t{}'.format(k, v)
                         for k, v in doc_reader.opt.items()]))

    while True:
        context = input('Context: ')
        question = input('Question: ')
        observation = {'text': '\n'.join([context, question]),
                       'episode_done': True}
        doc_reader.observe(observation)
        reply = doc_reader.act()
        print('Reply: %s' % reply['text'])


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
