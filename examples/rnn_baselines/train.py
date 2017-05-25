# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.agents.rnn_baselines.agents import Seq2SeqAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task

import torch

import copy
import os
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
    fn_suffix = opt['task'].lower()[:30]
    dict_tmp_fn = os.path.join(opt['logpath'], 'dict_{}.txt'.format(fn_suffix))
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

    model_fn = os.path.join(opt['logpath'], fn_suffix + '.model')
    if os.path.isfile(model_fn):
        print('Loading existing model parameters from ' + model_fn)
        agent.load(model_fn)

    opt['datatype'] = 'train'
    world_train = create_task(opt, agent)

    opt['datatype'] = 'valid'
    world_valid = create_task(opt, agent)

    start = time.time()
    # train / valid loop
    best_accuracy = 0
    with open(model_fn.replace('.model', '.validations'), 'w') as validations:
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

            # log validations and update best accuracy if applicable
            annotation = ''
            if report_valid['accuracy'] > best_accuracy:
                best_accuracy = report_valid['accuracy']
                agent.save(model_fn)
                annotation = '*' # mark this valid as a best one
            curr_time = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())
            validations.write('{}: {} {}\n'.format(
                curr_time, report_valid['accuracy'], annotation))
            validations.flush()
            report_valid['best_accuracy'] = best_accuracy
            print(report_valid)
            if report_valid['accuracy'] >= 1.0:
                break

    print('finished in {} s'.format(round(time.time() - start, 2)))


if __name__ == '__main__':
    main()
