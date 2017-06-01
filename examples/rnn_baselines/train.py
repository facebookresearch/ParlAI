# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.agents.rnn_baselines.seq2seq import Seq2seqAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
import parlai.core.build_data as bld

import torch

import copy
import os
import time

def main():
    # Get command line arguments
    parser = ParlaiParser(add_model_args=True)
    DictionaryAgent.add_cmdline_args(parser)
    Seq2seqAgent.add_cmdline_args(parser)
    parser.add_argument('--dict-maxexs', default=100000, type=int)
    opt = parser.parse_args()

    # set model_file if none set, default is based on task name
    if not opt['model_file']:
        logdir = os.path.join(opt['parlai_home'], 'logs')
        bld.make_dir(logdir)
        task_short = opt['task'].lower()[:30]
        opt['model_file'] = os.path.join(logdir, task_short + '.model')

    #
    opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
    if opt['cuda']:
        print('[ Using CUDA ]')
        torch.cuda.set_device(opt['gpu'])

    # set up dictionary
    print('Setting up dictionary.')
    if '.model' in opt['model_file']:
        dict_fn = opt['model_file'].replace('.model', '.dict')
    else:
        dict_fn = opt['model_file'] + '.dict'
    if os.path.isfile(dict_fn):
        opt['dict_loadpath'] = dict_fn
    dictionary = DictionaryAgent(opt)
    ordered_opt = copy.deepcopy(opt)
    cnt = 0

    # if dictionary was not loaded, create one
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
        dictionary.save(dict_fn, sort=True)

    # create agent
    agent = Seq2seqAgent(opt, {'dictionary': dictionary})

    if os.path.isfile(opt['model_file']):
        print('Loading existing model parameters from ' + opt['model_file'])
        agent.load(opt['model_file'])

    # create train and validation worlds
    opt['datatype'] = 'train'
    world_train = create_task(opt, agent)

    opt['datatype'] = 'valid'
    world_valid = create_task(opt, agent)

    # set up logging
    start = time.time()
    best_accuracy = 0
    if '.model' in opt['model_file']:
        valid_fn = opt['model_file'].replace('.model', '.validations')
        log_fn = opt['model_file'].replace('.model', '.log')
    else:
        valid_fn = opt['model_file'] + '.validations'
        log_fn = opt['model_file'] + '.log'

    # train / valid loop
    total = 0
    with open(valid_fn, 'w') as validations, open(log_fn, 'w') as log:
        while True:
            # train for a bit
            print('[ training ]')
            world_train.reset()
            for _ in range(200):
                world_train.parley()
                total += opt['batchsize']
            log.write('[ training example. ]\n')
            log.write(world_train.display() + '\n')

            # log training results
            print('[ training summary. ]')
            log.write('[ training summary. ]\n')
            report_train = world_train.report()
            report_train['cumulative_total'] = total
            print(report_train)
            log.write(str(report_train))
            log.write('\n')
            log.flush()

            # do one epoch of validation
            print('[ validating ]')
            world_valid.reset()
            for _ in world_valid:  # check valid accuracy
                world_valid.parley()
            log.write('[ validation example. ]\n')
            log.write(world_valid.display() + '\n')

            # get validation summary
            print('[ validation summary. ]')
            log.write('[ validation summary. ]\n')
            report_valid = world_valid.report()

            # update best accuracy if applicable
            annotation = ''
            if report_valid['accuracy'] > best_accuracy:
                best_accuracy = report_valid['accuracy']
                agent.save(opt['model_file'])
                annotation = '*'  # mark this validation as a best one
            curr_time = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())
            validations.write('{}: {} {}\n'.format(
                curr_time, report_valid['accuracy'], annotation))
            validations.flush()
            report_valid['best_accuracy'] = best_accuracy

            # log validation summary
            print(report_valid)
            log.write(str(report_valid))
            log.write('\n')
            log.flush()

            # break if accuracy reaches ~100%
            if report_valid['accuracy'] > 99.5:
                break

    print('finished in {} s'.format(round(time.time() - start, 2)))


if __name__ == '__main__':
    main()
