# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.agents.rnn_baselines.agents import Seq2SeqAgent
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task

from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch

import copy
import os
import random
import time

def main():
    # Get command line arguments
    parser = ParlaiParser()
    DictionaryAgent.add_cmdline_args(parser)
    Seq2SeqAgent.add_cmdline_args(parser)
    parser.add_argument('--dict-maxexs', default=100000, type=int)
    opt = parser.parse_args()

    opt['cuda'] = opt['cuda'] and torch.cuda.is_available()
    if opt['cuda']:
        print('[ Using CUDA ]')
        torch.cuda.set_device(opt['gpu'])

    # set up dictionary
    print('Setting up dictionary.')
    dict_tmp_fn = '/tmp/dict_{}.txt'.format(opt['task'])
    if os.path.isfile(dict_tmp_fn):
        opt['dict_loadpath'] = dict_tmp_fn
    dictionary = DictionaryAgent(opt)
    ordered_opt = copy.deepcopy(opt)
    cnt = 0

    if not opt.get('dict_loadpath'):
        for datatype in ['train:ordered', 'valid']:
            # we use train and valid sets to build dictionary
            ordered_opt['datatype'] = datatype
            ordered_opt['numthreads'] = 1
            ordered_opt['batchsize'] = 1
            world_dict = create_task(ordered_opt, dictionary)

            # pass examples to dictionary
            for _ in world_dict:
                cnt += 1
                if cnt > opt['dict_maxexs'] and opt['dict_maxexs'] > 0:
                    print('Processed {} exs, moving on.'.format(
                          opt['dict_maxexs']))
                    # don't wait too long...
                    break
                world_dict.parley()
        dictionary.save(dict_tmp_fn, sort=True)

    agent = Seq2SeqAgent(opt, {'dictionary': dictionary})

    opt['datatype'] = 'train'
    world_train = create_task(opt, agent)

    opt['datatype'] = 'valid'
    world_valid = create_task(opt, agent)

    start = time.time()
    # train / valid loop
    while True:
        print('[ training ]')
        for _ in range(200):  # train for a bit
            world_train.parley()

        print('[ training summary. ]')
        print(world_train.report())

        print('[ validating ]')
        world_valid.reset()
        for _ in world_valid:  # check valid accuracy
            world_valid.parley()

        print('[ validation summary. ]')
        report_valid = world_valid.report()
        print(report_valid)
        if report_valid['accuracy'] > 0.95:
            break

    print('finished in {} s'.format(round(time.time() - start, 2)))


if __name__ == '__main__':
    main()
