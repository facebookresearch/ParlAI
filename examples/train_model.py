# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train a model.
After training, computes validation and test error.
Run with, e.g.:
python examples/train_model.py -m ir_baseline -t dialog_babi:Task:1 -mf "/tmp/model"

TODO List:
- Add model specific params
- Validate & Log while training
- Keep best model from validation error, if desired
"""

from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer


def run_eval(agent, opt, datatype):
    ''' Eval on validation/test data. '''
    print("[running eval: " + datatype + "]")
    opt['datatype'] = datatype
    valid_world = create_task(opt, agent)
    for _ in range(len(valid_world)):
        valid_world.parley()
        if opt['display_examples']:
            print(valid_world.display() + "\n~~")
            print(valid_world.report())
        if valid_world.epoch_done():
            break
    valid_world.shutdown()
    metrics = datatype + ":" + str(valid_world.report())
    print(metrics)
    # Write out metrics
    if opt['model_file']:
        f = open(opt['model_file'] + '.' + datatype, "a+")
        f.write(metrics + '\n')
        f.close()

def build_dict():
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


def main():
    # Get command line arguments
    parser = ParlaiParser(True, True)
    parser.add_argument('-d', '--display-examples',
                        type='bool', default=False)
    parser.add_argument('-e', '--num-epochs', default=1)
    parser.add_argument('-mtt', '--max-train-time',
                        type=float, default=float('inf'))
    parser.add_argument('--dict-maxexs', default=100000, type=int)
    opt = parser.parse_args()
    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)

    # possibly set up dictionary

    train_time = Timer()
    print("[training...]")
    for i in range(opt['num_epochs'] * len(world)):
        world.parley()
        if opt['display_examples']:
            print(world.display() + "\n~~")
        if train_time.time() > opt['max_train_time']:
            print("[max_train_time elapsed: " + str(train_time.time()) + "]")
            break
    world.shutdown()

    if opt['model_file']:
        agent.save(opt['model_file'])
    run_eval(agent, opt, 'valid')
    run_eval(agent, opt, 'test')


if __name__ == '__main__':
    main()


