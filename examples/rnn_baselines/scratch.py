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
    # create agent
    agent = Seq2seqAgent(opt, {'dictionary': dictionary})

    if os.path.isfile(opt['model_file']):
        print('Loading existing model parameters from ' + opt['model_file'])
        agent.load(opt['model_file'])


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
