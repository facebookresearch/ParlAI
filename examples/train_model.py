# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train a model. 

After training, computes validation and test error.

Run with, e.g.:

python examples/train_model.py -m ir_baseline -t dialog_babi:Task:1 -mf "/tmp/model"

..or..

python examples/train_model.py -m rnn_baselines/seq2seq -t babi:Task1k:1 -mf "/tmp/model" -dbf True -bs 32 -lr 0.5 -hs 128

..or..

rm -f /tmp/modelz; python examples/train_model.py -m drqa -t babi:Task1k:1 -mf "/tmp/modelz" -dbf True

TODO List:
- More logging (e.g. to files), make things prettier.
"""

from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent
from parlai.agents.drqa.drqa import SimpleDictionaryAgent
from parlai.core.utils import Timer
import copy
import math
import os

def run_eval(agent, opt, datatype, still_training=False):
    ''' Eval on validation/test data. '''
    print("[running eval: " + datatype + "]")
    opt['datatype'] = datatype
    valid_world = create_task(opt, agent)
    for i in range(len(valid_world)):
        valid_world.parley()
        if i == 1 and opt['display_examples']:
            print(valid_world.display() + "\n~~")
            print(valid_world.report())
        if valid_world.epoch_done():
            break
    valid_world.shutdown()
    valid_report = valid_world.report()
    metrics = datatype + ":" + str(valid_report)
    print(metrics)
    if still_training:
        return valid_report
    else:
        if opt['model_file']:
            # Write out metrics
            f = open(opt['model_file'] + '.' + datatype, "a+")
            f.write(metrics + '\n')
            f.close()

def build_dict(opt):
    print('[setting up dictionary.]')
    if 'dict_loadpath' not in opt:
        if '.model' in opt['model_file']:
            dict_fn = opt['model_file'].replace('.model', '.dict')
        else:
            dict_fn = opt['model_file'] + '.dict'
        opt['dict_loadpath'] = dict_fn
    if os.path.isfile(opt['dict_loadpath']):
        # dict already built
        print("[dict already built.]")
        return
    opt['dict_savepath'] = opt['dict_loadpath']
    opt.pop('dict_loadpath', None)
    dictionary = SimpleDictionaryAgent(opt)
    ordered_opt = copy.deepcopy(opt)
    cnt = 0
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
    opt['dict_loadpath'] = opt['dict_savepath']
    opt.pop('dict_savepath', None)
    print('[dictionary built.]')
    print('[num words =  %d]' % len(dictionary))

def main():
    # Get command line arguments
    parser = ParlaiParser(True, True)
    parser.add_argument('-d', '--display-examples',
                        type='bool', default=False)
    parser.add_argument('-e', '--num-epochs', type=int, default=1)
    parser.add_argument('-ttim', '--max-train-time',
                        type=float, default=float('inf'))
    parser.add_argument('-ltim', '--log-every-n-secs',
                        type=float, default=1)
    parser.add_argument('-vtim', '--validate-every-n-secs',
                        type=float, default=False)
    parser.add_argument('-dbf', '--dict_build_first',
                        type='bool', default=False,
                        help='build dictionary first before training agent')
    opt = parser.parse_args()
    # Possibly build a dictionary (not all models do this).
    if opt['dict_build_first']:
        build_dict(opt)
    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)

    train_time = Timer()
    validate_time = Timer()
    log_time = Timer()
    print("[training...]")
    parleys = 0
    num_parleys = opt['num_epochs'] * len(world)
    best_accuracy = 0
    for i in range(num_parleys):
        world.parley()
        parleys = parleys + 1
        if train_time.time() > opt['max_train_time']:
            print("[max_train_time elapsed: " + str(train_time.time()) + "]")
            break
        if log_time.time() > opt['log_every_n_secs']:
            if opt['display_examples']:
                print(world.display() + "\n~~")
            parleys_per_sec =  train_time.time() / parleys
            time_left = (num_parleys - parleys) * parleys_per_sec
            log = ("[time:" + str(math.floor(train_time.time()))
                  + "s parleys:" + str(parleys) 
                  + " time_left:"
                  + str(math.floor(time_left))  + "s] ")
            if hasattr(agent, 'report'):
                log = log + (agent.report())
            else:
                log = log + (world.report())
                # TODO: world.reset_metrics()
            print(log)
            log_time.reset()
        if (opt['validate_every_n_secs'] and
            validate_time.time() > opt['validate_every_n_secs']):
            valid_report = run_eval(agent, opt, 'valid', True)
            if valid_report['accuracy'] > best_accuracy:
                best_accuracy = valid_report['accuracy']
                print("[current best accuracy: " + str(best_accuracy) +  "]")
                if opt['model_file']:
                    agent.save(opt['model_file'])
            validate_time.reset()
    world.shutdown()

    if opt['model_file']:
        agent.save(opt['model_file'])
    run_eval(agent, opt, 'valid')
    run_eval(agent, opt, 'test')


if __name__ == '__main__':
    main()


