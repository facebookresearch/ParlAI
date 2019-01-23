#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Given a specified task, builds a dictionary on the training and validation
set for that task and then trains a memory network using the default parameters
on that task.
This example uses the ParsedRemoteAgent, which does all of the parsing in
python so that the parsing is done by exactly the same python code both in the
building of the dictionary and the train/test-time parsing.
Alternatively, a regular RemoteAgent could be used, which implements its own
parsing (and could also build its own dictionary).
"""

from parlai.agents.remote_agent.remote_agent import ParsedRemoteAgent
from parlai.core.worlds import create_task
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser

import copy
import os
import sys
import time


def main():
    # Get command line arguments
    argparser = ParlaiParser()
    DictionaryAgent.add_cmdline_args(argparser)
    ParsedRemoteAgent.add_cmdline_args(argparser)
    argparser.add_argument('--num-examples', default=1000, type=int)
    argparser.add_argument('--num-its', default=100, type=int)
    argparser.add_argument('--dict-max-exs', default=10000, type=int)
    parlai_home = os.environ['PARLAI_HOME']
    if '--remote-cmd' not in sys.argv:
        if os.system('which luajit') != 0:
            raise RuntimeError('Could not detect torch luajit installed: ' +
                               'please install torch from http://torch.ch ' +
                               'or manually set --remote-cmd for this example.')
        sys.argv.append('--remote-cmd')
        sys.argv.append('luajit {}/parlai/agents/legacy_agents/'.format(
            parlai_home) + 'memnn_luatorch_cpu/memnn_zmq_parsed.lua')
    if '--remote-args' not in sys.argv:
        sys.argv.append('--remote-args')
        sys.argv.append('{}/examples/'.format(parlai_home) +
                        'memnn_luatorch_cpu/params_default.lua')

    opt = argparser.parse_args()

    # set up dictionary
    print('Setting up dictionary.')
    dictionary = DictionaryAgent(opt)
    if not opt.get('dict_file'):
        # build dictionary since we didn't load it
        ordered_opt = copy.deepcopy(opt)
        ordered_opt['datatype'] = 'train:ordered'
        ordered_opt['numthreads'] = 1
        world_dict = create_task(ordered_opt, dictionary)

        print('Dictionary building on training data.')
        cnt = 0
        # pass examples to dictionary
        while not world_dict.epoch_done():
            cnt += 1
            if cnt > opt['dict_max_exs'] and opt['dict_max_exs'] > 0:
                print('Processed {} exs, moving on.'.format(
                      opt['dict_max_exs']))
                # don't wait too long...
                break

            world_dict.parley()

        # we need to save the dictionary to load it in memnn (sort it by freq)
        dictionary.sort()
        dictionary.save('/tmp/dict.txt', sort=True)

    print('Dictionary ready, moving on to training.')

    opt['datatype'] = 'train'
    agent = ParsedRemoteAgent(opt, {'dictionary_shared': dictionary.share()})
    world_train = create_task(opt, agent)

    valid_opt = copy.deepcopy(opt)
    valid_opt['datatype'] = 'valid'
    # switch to 1 thread, the memnn code will handle it better
    valid_opt['numthreads'] = 1
    world_valid = create_task(valid_opt, agent)

    start = time.time()
    with world_train:
        for _ in range(opt['num_its']):
            print('[ training ]')
            for _ in range(opt['num_examples'] * opt.get('numthreads', 1)):
                world_train.parley()

            print('[ validating ]')
            world_valid.reset()
            while not world_valid.epoch_done():  # check valid accuracy
                world_valid.parley()

            print('[ validation summary. ]')
            report_valid = world_valid.report()
            print(report_valid)
            if report_valid['accuracy'] > 0.95:
                break

        # show some example dialogs after training:
        world_valid = create_task(valid_opt, agent)
        for _k in range(3):
            world_valid.parley()
            print(world_valid.display())

    print('finished in {} s'.format(round(time.time() - start, 2)))


if __name__ == '__main__':
    main()
