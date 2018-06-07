# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which allows local human keyboard input to talk to a trained model.

For example:
`python examples/interactive.py -m drqa -mf "models:drqa/squad/model"`

Then enter something like:
"Bob is Blue.\nWhat is Bob?"
as the user input (or in general for the drqa model, enter
a context followed by '\n' followed by a question all as a single input.)
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.local_human.local_human import LocalHumanAgent

import random

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('--display-prettify', type='bool', default=False,
                        help='Set to use a prettytable when displaying '
                             'examples with text candidates')
    parser.add_argument('--display-ignore-fields',type=str,
                        default='label_candidates,text_candidates',
                        help='Do not display these fields')
    LocalHumanAgent.add_cmdline_args(parser)
    return parser


def interactive(opt):
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: interactive should be passed opt not Parser ]')
        opt = opt.parse_args()
    opt['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)
    
    # Show some example dialogs:
    while True:
        world.parley()
        if opt.get('display_examples'):
            print("---")
            print(world.display())
        if world.epoch_done():
            print("EPOCH DONE")
            break


if __name__ == '__main__':
    random.seed(42)
    interactive(setup_args().parse_args())
