#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic example which iterates through the tasks specified and prints them out. Used for
verification of data loading and iteration.

For example, to make sure that bAbI task 1 (1k exs) loads one can run and to
see a few of them:

Examples
--------

.. code-block:: shell

  python display_data.py -t babi:task1k:1
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger

import random
import io

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Display data from a task')
    parser.add_pytorch_datateacher_args()
    # Get command line arguments
    parser.add_argument('-n', '-ne', '--num-examples', type=int, default=10)
    parser.add_argument('-mdl', '--max-display-len', type=int, default=1000)
    parser.add_argument('--display-ignore-fields', type=str, default='agent_reply')
    parser.set_defaults(datatype='train:stream')
    return parser


def clean(t):
    t = t[2:-2]
    if t[0] == ' ':
        t = t[1:]
    t = t.rstrip(' ')
    i1 = t.find("''")
    i2 = t.find("``")
    if i1 < i2:
        #print(t)
        t = t[0:i1] + t[i2+2:].lstrip(' ')
        #print(t)
        #import pdb; pdb.set_trace()
    return t
    

def display_data(opt):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    # set up logging
    log_every_n_secs = 1
    log_time = TimeLogger()
    
    cands = {}
    
    # Show some example dialogs.
    for _ in range(opt['num_examples']):
        world.parley()

        if log_time.time() > log_every_n_secs:
            report = world.report()
            text, report = log_time.log(report['exs'], world.num_examples(), report)
            print(text +  "cands: " + str(len(cands)))
        
        txt = world.acts[0]['text']
        for t in txt.split('\n')[:-1]:
            if t.startswith("``") and t.endswith("''"):
                t = clean(t)
                if t not in cands:
                    cands[t] = True
                    #print(t)
                    #if t.find('XXXX') != -1:
                    #    print(t)

        if world.epoch_done():
            print('EPOCH DONE')
            break

    try:
        # print dataset size if available
        print(
            '[ loaded {} episodes with a total of {} examples ]'.format(
                world.num_episodes(), world.num_examples()
            )
        )
    except Exception:
        pass

    print("cands: " + str(len(cands)))
    f = io.open("/tmp/cands.txt", "w")
    for c in cands:
        f.write(c + '\n')
    f.close()

if __name__ == '__main__':
    random.seed(42)

    # Get command line arguments
    parser = setup_args()
    opt = parser.parse_args()
    display_data(opt)
