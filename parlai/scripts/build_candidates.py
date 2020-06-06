#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Build the candidate responses for a retrieval model.

Examples
--------

.. code-block:: shell

  python build_candidates.py -t convai2 --outfile /tmp/cands.txt
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
import parlai.utils.logging as logging
import random
import tempfile


def build_cands(opt):
    # create repeat label agent and assign it to the specified task
    if opt['numthreads'] > 1:
        # Broken in hogwild mode. Just fall back to single processing mode
        opt['numthreads'] = 1
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    if opt['outfile'] is None:
        outfile = tempfile.mkstemp(
            prefix='{}_{}_'.format(opt['task'], opt['datatype']), suffix='.txt'
        )[1]
    else:
        outfile = opt['outfile']

    if opt.get('num_examples', -1) == -1:
        num_examples = world.num_examples()
    else:
        num_examples = opt['num_examples']
    log_timer = TimeLogger()

    logging.info(f'Starting to build candidates from task.. (ex: {num_examples})')
    logging.info(f'Saving output to {outfile}')
    cands = set()
    for _ in range(num_examples):
        world.parley()
        # We get the acts of the first agent, which is the teacher.
        acts = world.get_acts()[0]
        if isinstance(acts, dict):
            # We turn into a batch of 1 example, in case batching is being used.
            acts = [acts]
        for a in acts:
            candidate = a.get('labels', a.get('eval_labels', None))
            if candidate is not None:
                candidate = candidate[0]
                cands.add(candidate)
        if log_timer.time() > opt['log_every_n_secs']:
            text, _log = log_timer.log(world.total_parleys, world.num_examples())
            logging.info(text)
        if world.epoch_done():
            logging.info('epoch done')
            break
    fw = open(outfile, 'w')
    fw.write('\n'.join(cands))
    fw.close()


def main():
    random.seed(42)
    # Get command line arguments
    parser = ParlaiParser()
    parser.add_argument(
        '-n',
        '--num-examples',
        default=-1,
        type=int,
        help='Total number of exs to convert, -1 to convert all examples',
    )
    parser.add_argument(
        '-of',
        '--outfile',
        default=None,
        type=str,
        help='Output file where to save, by default will be created in /tmp',
    )
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.set_defaults(datatype='train:evalmode')
    opt = parser.parse_args()
    build_cands(opt)


if __name__ == '__main__':
    main()
