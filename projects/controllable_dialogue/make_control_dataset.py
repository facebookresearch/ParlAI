#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Make a copy of the ConvAI2 dataset with CT control variables annotated.
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import msg_to_str, TimeLogger
from controllable_seq2seq.util import ConvAI2History
from controllable_seq2seq.controls import eval_attr, initialize_control_information
import random


def make_dataset(opt):

    # Initialize control information so we can compute sentence attributes.
    # Here we set build_task=False so we don't download data/controllable_dialogue
    # (because we're trying to create it instead).
    initialize_control_information(opt, build_task=False)

    # Create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    ignorefields = opt.get('ignore_fields', '')
    outfile = opt['outfile']

    # Number of examples to process
    if opt['num_examples'] == -1:
        num_examples = world.num_examples()
    else:
        num_examples = opt['num_examples']

    # List of controls to include:
    controls = opt['controls'].split(',') if opt['controls'] != '' else []

    print('[ starting to convert.. ]')
    print('[ saving output to {} ]'.format(outfile))
    fw = open(outfile, 'w')
    log_timer = TimeLogger()

    for _ in range(num_examples):
        world.parley()
        world.acts[0]['labels'] = world.acts[0].get(
            'labels', world.acts[0].pop('eval_labels', None)
        )

        # Need to get history in order to compute control values
        hist = ConvAI2History(world.acts[0]['text'], assume_persontokens=False)
        response = world.acts[0]['labels'][0]

        # Compute control values
        for ctrl in controls:
            ctrl_val = eval_attr(response, hist, ctrl)
            if ctrl == 'avg_nidf':
                assert ctrl_val >= 0
                assert ctrl_val <= 1
            elif ctrl == 'question':
                assert ctrl_val in [0, 1]
            elif ctrl == 'lastuttsim':
                if ctrl_val is not None:
                    assert ctrl_val >= -1
                    assert ctrl_val <= 1
            else:
                raise Exception('unexpected ctrl name: %s' % ctrl)
            world.acts[0][ctrl] = ctrl_val  # add control value to act

        # Write to file
        txt = msg_to_str(world.acts[0], ignore_fields=ignorefields)
        fw.write(txt + '\n')
        if world.acts[0].get('episode_done', False):
            fw.write('\n')

        if log_timer.time() > opt['log_every_n_secs']:
            text, _log = log_timer.log(world.total_parleys, world.num_examples())
            print(text)

        if world.epoch_done():
            print('EPOCH DONE')
            break
    fw.close()


if __name__ == '__main__':
    random.seed(42)
    # Get command line arguments
    parser = ParlaiParser()
    parser.add_argument(
        '-n',
        '--num-examples',
        default=-1,
        type=int,
        help='Total number of exs to convert, -1 to convert \
                                all examples',
    )
    parser.add_argument(
        '-of',
        '--outfile',
        default=None,
        type=str,
        help='Output file where to save, by default will be \
                                created in /tmp',
    )
    parser.add_argument(
        '-if',
        '--ignore-fields',
        default='id',
        type=str,
        help='Ignore these fields from the message (returned\
                                with .act() )',
    )
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)

    parser.add_argument(
        '--controls',
        type=str,
        default='',
        help='comma-separated controls to be included',
    )

    parser.set_defaults(task="fromfile:parlaiformat")
    parser.set_defaults(datatype="train:stream")

    opt = parser.parse_args()
    make_dataset(opt)
