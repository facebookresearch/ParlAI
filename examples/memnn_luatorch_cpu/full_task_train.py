# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Given a specified task, builds a dictionary on the training and validation
set for that task and then trains a memory network using the default parameters
on that task.
This example uses the ParsedRemoteAgent, which does all of the parsing in
python so that the parsing is done by exactly the same python code both in the
building of the dictionary and the train/test-time parsing.
Alternatively, a regular RemoteAgent could be used, which implements its own
parsing (and could also build its own dictionary).
"""

from parlai.agents.remote_agent.agents import ParsedRemoteAgent
from parlai.core.worlds import create_task
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import DialogPartnerWorld, HogwildWorld

import copy
import sys
import time


def main():
    # Get command line arguments
    argparser = ParlaiParser()
    DictionaryAgent.add_cmdline_args(argparser)
    ParsedRemoteAgent.add_cmdline_args(argparser)
    if '--remote-cmd' not in sys.argv:
        sys.argv.append('--remote-cmd')
        sys.argv.append('luajit parlai/agents/memnn_luatorch_cpu/' +
                        'memnn_zmq_parsed.lua')
    if '--remote-args' not in sys.argv:
        sys.argv.append('--remote-args')
        sys.argv.append('examples/memnn_luatorch_cpu/params/params_default.lua')

    opt = argparser.parse_args()

    # set up dictionary
    print('Setting up dictionary.')
    dictionary = DictionaryAgent(opt)
    if not opt.get('dict_loadpath'):
        # build dictionary since we didn't load it
        ordered_opt = copy.deepcopy(opt)
        for datatype in ['train:ordered', 'valid']:
            # we use train and valid sets to build dictionary
            ordered_opt['datatype'] = datatype
            ordered_opt['numthreads'] = 1
            world_dict = create_task(ordered_opt, dictionary)
            # pass examples to dictionary
            for _ in world_dict:
                world_dict.parley()

        # we need to save the dictionary to load it in memnn (sort it by freq)
        dictionary.save('/tmp/dict.txt', sort=True)

    print('Dictionary ready, moving on to training.')

    opt['datatype'] = 'train'
    agent = ParsedRemoteAgent(opt, {'dictionary': dictionary})
    world_train = create_task(opt, agent)
    opt['datatype'] = 'valid'
    world_valid = create_task(opt, agent)

    start = time.time()
    with world_valid, world_train:
        for _ in range(100):
            print('[ training ]')
            for _ in range(1000 * opt.get('numthreads', 1)):
                world_train.parley()
            world_train.synchronize()

            print('[ training summary. ]')
            print(world_train.report())

            print('[ validating ]')
            for _ in world_valid:  # check valid accuracy
                world_valid.parley()

            print('[ validation summary. ]')
            report_valid = world_valid.report()
            print(report_valid)
            if report_valid['accuracy'] > 0.95:
                break

        # show some example dialogs after training:
        for _k in range(3):
            world_valid.parley()
            print(world_valid.display())

    print('finished in {} s'.format(round(time.time() - start, 2)))

if __name__ == '__main__':
    main()
