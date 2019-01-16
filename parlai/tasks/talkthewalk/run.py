#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Runs Talk the Walk Simulation using pretrained model files for tourist and
agent.
"""

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.logs import TensorboardLogger
from parlai.core.worlds import BatchWorld, HogwildWorld
from parlai.tasks.talkthewalk.worlds import SimulateWorld

import copy
import os

from build import build


def _path(opt):
    build(opt)
    opt['ttw_data'] = os.path.join(opt['datapath'], 'TalkTheWalk')


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Evaluate a model')
    parser.add_parlai_data_path()
    # Get command line arguments
    parser.add_argument('-tmf', '--tourist-model-file', type=str)
    parser.add_argument('-gmf', '--guide-model-file', type=str)
    parser.add_argument('-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('--metrics', type=str, default="all",
                        help="list of metrics to show/compute, e.g. "
                             "ppl,f1,accuracy,hits@1."
                             "If 'all' is specified [default] all are shown.")
    TensorboardLogger.add_cmdline_args(parser)
    parser.set_defaults(datatype='valid')
    return parser


def run(opt):
    opt = copy.deepcopy(opt)
    _path(opt)

    opt['model_file'] = opt['tourist_model_file']
    tourist = create_agent(opt)
    opt['model_file'] = opt['guide_model_file']
    guide = create_agent(opt)

    world = SimulateWorld(opt, [tourist, guide])

    if opt.get('numthreads', 1) > 1:
        # use hogwild world if more than one thread requested
        # hogwild world will create sub batch worlds as well if bsz > 1
        world = HogwildWorld(opt, world)
    elif opt.get('batchsize', 1) > 1:
        # otherwise check if should use batchworld
        world = BatchWorld(opt, world)

    # Show some example dialogs:
    cnt = 0
    while not world.epoch_done():
        cnt += opt.get('batchsize', 1)
        world.parley()
        if opt['display_examples']:
            print(world.display() + "\n~~")
        if opt['num_examples'] > 0 and cnt >= opt['num_examples']:
            break

    if world.epoch_done():
        print("EPOCH DONE")
    print('finished evaluating task using datatype {}'.format(
          opt.get('datatype', 'N/A')))
    report = world.report()
    print(report)
    return report


if __name__ == '__main__':
    parser = setup_args()
    run(parser.parse_args(print_args=False))
