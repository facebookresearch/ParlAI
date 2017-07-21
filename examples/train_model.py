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
from parlai.core.utils import Timer, TensorboardLogger
import build_dict
import copy
import importlib
import math
import os


def run_eval(agent, opt, datatype, still_training=False, max_exs=-1):
    ''' Eval on validation/test data. '''
    print('[ running eval: ' + datatype + ' ]')
    opt['datatype'] = datatype
    if opt.get('evaltask'):
        opt['task'] = opt['evaltask']

    valid_world = create_task(opt, agent)
    cnt = 0
    for _ in valid_world:
        valid_world.parley()
        if cnt == 0 and opt['display_examples']:
            first_run = False
            print(valid_world.display() + '\n~~')
            print(valid_world.report())
        cnt += opt['batchsize']
        if valid_world.epoch_done() or (max_exs > 0 and cnt > max_exs):
            # note this max_exs is approximate--some batches won't always be
            # full depending on the structure of the data
            break
    valid_world.shutdown()
    valid_report = valid_world.report()

    metrics = datatype + ':' + str(valid_report)
    print(metrics)
    if still_training:
        return valid_report
    elif opt['model_file']:
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
    train.add_argument('-e', '--num-epochs', type=float, default=-1)
    train.add_argument('-ttim', '--max-train-time',
                        type=float, default=-1)
    train.add_argument('-ltim', '--log-every-n-secs',
                        type=float, default=2)
    train.add_argument('-vtim', '--validation-every-n-secs',
                        type=float, default=-1)
    train.add_argument('-vme', '--validation-max-exs',
                        type=int, default=-1,
                        help='max examples to use during validation (default ' +
                             '-1 uses all)')
    train.add_argument('-vp', '--validation-patience',
                        type=int, default=5,
                        help=('number of iterations of validation where result '
                              + 'does not improve before we stop training'))
    train.add_argument('-dbf', '--dict-build-first',
                        type='bool', default=True,
                        help='build dictionary first before training agent')
    train.add_argument('--tensorboard-dir', default='',
                        help='directory for tensorboard logging')
    opt = parser.parse_args()
    # Possibly build a dictionary (not all models do this).
    if opt['dict_build_first'] and 'dict_file' in opt:
        if opt['dict_file'] is None and opt.get('model_file'):
            opt['dict_file'] = opt['model_file'] + '.dict'
        print("[ building dictionary first... ]")
        build_dict.build_dict(opt)

    # Set up logger if directory provided
    if opt['tensorboard_dir']:
        logger = TensorboardLogger(opt['tensorboard_dir'])
        opt['train_log'] = {}
        opt['valid_log'] = {}

    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)

    train_time = Timer()
    validate_time = Timer()
    log_time = Timer()
    print('[ training... ]')
    parleys = 0
    total_exs = 0
    max_exs = opt['num_epochs'] * len(world)
    max_parleys = math.ceil(max_exs / opt['batchsize'])
    best_accuracy = 0
    impatience = 0
    saved = False
    while True:
        world.parley()
        parleys += 1

        if opt['num_epochs'] > 0 and parleys >= max_parleys:
            print('[ num_epochs completed: {} ]'.format(opt['num_epochs']))
            break
        if opt['max_train_time'] > 0 and train_time.time() > opt['max_train_time']:
            print('[ max_train_time elapsed: {} ]'.format(train_time.time()))
            break
        if opt['log_every_n_secs'] > 0 and log_time.time() > opt['log_every_n_secs']:
            if opt['display_examples']:
                print(world.display() + '\n~~')

            logs = []
            # time elapsed
            logs.append('time:{}s'.format(math.floor(train_time.time())))
            logs.append('parleys:{}'.format(parleys))

            # get report and update total examples seen so far
            if hasattr(agent, 'report'):
                train_report = agent.report()
                agent.reset_metrics()
            else:
                train_report = world.report()
                world.reset_metrics()

            if hasattr(train_report, 'get') and train_report.get('total'):
                total_exs += train_report['total']
                logs.append('total_exs:{}'.format(total_exs))

            # check if we should log amount of time remaining
            time_left = None
            if opt['num_epochs'] > 0:
                exs_per_sec = train_time.time() / total_exs
                time_left = (max_exs - total_exs) * exs_per_sec
            if opt['max_train_time'] > 0:
                other_time_left = opt['max_train_time'] - train_time.time()
                if time_left is not None:
                    time_left = min(time_left, other_time_left)
                else:
                    time_left = other_time_left
            if time_left is not None:
                logs.append('time_left:{}s'.format(math.floor(time_left)))

            # join log string and add full metrics report to end of log
            log = '[ {} ] {}'.format(' '.join(logs), train_report)

            print(log)
            log_time.reset()

            # Log train values
            if opt['tensorboard_dir']:
                logger.log_train(train_report, total_exs)
                logger.log_train(opt['train_log'], total_exs)

        if (opt['validation_every_n_secs'] > 0 and
                validate_time.time() > opt['validation_every_n_secs']):
            valid_report = run_eval(agent, opt, 'valid', True, opt['validation_max_exs'])

            # Log valid values
            if opt['tensorboard_dir']:
                logger.log_valid(valid_report, total_exs)
                logger.log_valid(opt['valid_log'], total_exs)

            if valid_report['accuracy'] > best_accuracy:
                best_accuracy = valid_report['accuracy']
                impatience = 0
                print('[ new best accuracy: ' + str(best_accuracy) +  ' ]')
                world.save_agents()
                saved = True
                if best_accuracy == 1:
                    print('[ task solved! stopping. ]')
                    break
            else:
                impatience += 1
                print('[ did not beat best accuracy: {} impatience: {} ]'.format(
                        round(best_accuracy, 4), impatience))
            validate_time.reset()
            if opt['validation_patience'] > 0 and impatience >= opt['validation_patience']:
                print('[ ran out of patience! stopping training. ]')
                break
    world.shutdown()
    if not saved:
        world.save_agents()
    else:
        # reload best validation model
        agent = create_agent(opt)

    run_eval(agent, opt, 'valid')
    run_eval(agent, opt, 'test')


if __name__ == '__main__':
    main()
