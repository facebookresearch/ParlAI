# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import create_agent
from parlai.core.build_data import download_models
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer, round_sigfigs
from parlai.core.worlds import create_task
from .build_dict import build_dict
from .worlds import PerplexityWorld


def eval_ppl(parser):
    """Evaluates the the perplexity and f1 of a model (and hits@1 if model has
    ranking enabled.
    """
    dict_agent = build_dict()

    opt = parser.parse_args()

    # create agents
    agent = create_agent(opt)
    world = create_task(opt, [agent, dict_agent], default_world=PerplexityWorld)
    world.dict = dict_agent

    # set up logging
    log_time = Timer()
    tot_time = 0

    # Show some example dialogs:
    while not world.epoch_done():
        world.parley()

        if log_time.time() > 1:  # log every 1 sec
            tot_time += log_time.time()
            report = world.report()
            print('{}s elapsed, {}%% complete, {}'.format(
                int(tot_time),
                round_sigfigs(report['total'] / world.num_examples() * 100, 2),
                report))
            log_time.reset()
    if world.epoch_done():
        print('EPOCH DONE')
    tot_time += log_time.time()
    final_report = world.report()
    print('{}s elapsed: {}'.format(int(tot_time), final_report))

if __name__ == '__main__':
    parser = ParlaiParser(True, True)
    eval_ppl(parser)
