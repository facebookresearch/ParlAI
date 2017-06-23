# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
'''Train a model.

After training, computes validation and test error.

Run with, e.g.:

python examples/train_model.py -m ir_baseline -t dialog_babi:Task:1 -mf '/tmp/model'

..or..

python examples/train_model.py -m rnn_baselines/seq2seq -t babi:Task10k:1 -mf '/tmp/model' -bs 32 -lr 0.5 -hs 128

..or..

python examples/train_model.py -m drqa -t babi:Task10k:1 -mf '/tmp/model' -bs 10

TODO List:
- More logging (e.g. to files), make things prettier.
'''

from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer
import build_dict
import copy
import importlib
import math
import os

def run_eval(agent, opt, datatype, still_training=False):
    ''' Eval on validation/test data. '''
    print('[ running eval: ' + datatype + ' ]')
    opt['datatype'] = datatype
    if opt.get('evaltask'):
        opt['task'] = opt['evaltask']
    valid_world = create_task(opt, agent)
    for i in range(len(valid_world)):
        valid_world.parley()
        if i == 1 and opt['display_examples']:
            print(valid_world.display() + '\n~~')
            print(valid_world.report())
        if valid_world.epoch_done():
            break
    valid_world.shutdown()
    valid_report = valid_world.report()
    metrics = datatype + ':' + str(valid_report)
    print(metrics)
    if still_training:
        return valid_report
    else:
        if opt['model_file']:
            # Write out metrics
            f = open(opt['model_file'] + '.' + datatype, 'a+')
            f.write(metrics + '\n')
            f.close()

def main():
    # Get command line arguments
    parser = ParlaiParser(True, True)
    train = parser.add_argument_group('Training Loop Arguments')
    train.add_argument('-et', '--evaltask',
                        help=('task to use for valid/test (defaults to the ' +
                              'one used for training if not set)'))
    train.add_argument('-d', '--display-examples',
                        type='bool', default=False)
    train.add_argument('-e', '--num-epochs', type=int, default=1)
    train.add_argument('-ttim', '--max-train-time',
                        type=float, default=float('inf'))
    train.add_argument('-ltim', '--log-every-n-secs',
                        type=float, default=1)
    train.add_argument('-vtim', '--validation-every-n-secs',
                        type=float, default=0)
    train.add_argument('-vp', '--validation-patience',
                        type=int, default=5,
                        help=('number of iterations of validation where result '
                              + 'does not improve before we stop training'))
    train.add_argument('-dbf', '--dict-build-first',
                        type='bool', default=True,
                        help='build dictionary first before training agent')
    opt = parser.parse_args()
    # Possibly build a dictionary (not all models do this).
    if opt['dict_build_first'] and 'dict_file' in opt:
        if opt['dict_file'] is None and opt.get('model_file'):
            opt['dict_file'] = opt['model_file'] + '.dict'
        build_dict.build_dict(opt)
    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)

    train_time = Timer()
    validate_time = Timer()
    log_time = Timer()
    print('[ training... ]')
    parleys = 0
    num_parleys = opt['num_epochs'] * int(len(world) / opt['batchsize'])
    best_accuracy = 0
    impatience = 0
    saved = False
    for i in range(num_parleys):
        world.parley()
        parleys = parleys + 1
        if train_time.time() > opt['max_train_time']:
            print('[ max_train_time elapsed: ' + str(train_time.time()) + ' ]')
            break
        if log_time.time() > opt['log_every_n_secs']:
            if opt['display_examples']:
                print(world.display() + '\n~~')
            parleys_per_sec =  train_time.time() / parleys
            time_left = (num_parleys - parleys) * parleys_per_sec
            log = ('[ time:' + str(math.floor(train_time.time()))
                  + 's parleys:' + str(parleys)
                  + ' time_left:'
                  + str(math.floor(time_left))  + 's ]')
            if hasattr(agent, 'report'):
                log = log + str(agent.report())
            else:
                log = log + str(world.report())
                # TODO: world.reset_metrics()
            print(log)
            log_time.reset()
        if (opt['validation_every_n_secs'] and
            validate_time.time() > opt['validation_every_n_secs']):
            valid_report = run_eval(agent, opt, 'valid', True)
            if valid_report['accuracy'] > best_accuracy:
                best_accuracy = valid_report['accuracy']
                impatience = 0
                print('[ new best accuracy: ' + str(best_accuracy) +  ' ]')
                if opt['model_file']:
                    agent.save(opt['model_file'])
                    saved = True
                if best_accuracy == 1:
                    print('[ task solved! stopping. ]')
                    break
            else:
                impatience += 1
                print('[ did not beat best accuracy: ' + str(best_accuracy) +
                      ' impatience: ' + str(impatience)  + ' ]')
            validate_time.reset()
            if impatience >= opt['validation_patience']:
                print('[ ran out of patience! stopping. ]')
                break
    world.shutdown()
    if not saved:
        if opt['model_file']:
            agent.save(opt['model_file'])
    else:
        # reload best validation model
        agent = create_agent(opt)

    run_eval(agent, opt, 'valid')
    run_eval(agent, opt, 'test')


if __name__ == '__main__':
    main()
