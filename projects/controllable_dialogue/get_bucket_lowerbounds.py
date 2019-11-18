#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
For a given (continuous) control variable in the dataset, bucket the data and return the
lower bounds for those buckets.
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from controllable_seq2seq.controls import sort_into_bucket
from collections import Counter
import random


def bucket_data(opt):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    if opt['num_examples'] == -1:
        num_examples = world.num_examples()
    else:
        num_examples = opt['num_examples']
    log_timer = TimeLogger()

    assert opt['control'] != ''
    ctrl = opt['control']

    num_buckets = opt['num_buckets']

    ctrl_vals = []  # list of floats

    for _ in range(num_examples):
        world.parley()
        world.acts[0]['labels'] = world.acts[0].get(
            'labels', world.acts[0].pop('eval_labels', None)
        )

        if ctrl not in world.acts[0].keys():
            raise Exception(
                'Error: control %s isn\'t in the data. available keys: %s'
                % (ctrl, ', '.join(world.acts[0].keys()))
            )
        ctrl_val = world.acts[0][ctrl]
        if ctrl_val == "None":
            assert ctrl == 'lastuttsim'
            ctrl_val = None
        else:
            ctrl_val = float(ctrl_val)
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
            raise Exception('Unexpected ctrl name: %s' % ctrl)
        ctrl_vals.append(ctrl_val)

        if log_timer.time() > opt['log_every_n_secs']:
            text, _log = log_timer.log(world.total_parleys, world.num_examples())
            print(text)

        if world.epoch_done():
            print('EPOCH DONE')
            break

    if ctrl == 'lastuttsim':
        num_nones = len([v for v in ctrl_vals if v is None])
        ctrl_vals = [v for v in ctrl_vals if v is not None]
        print(
            "Have %i Nones for lastuttsim; these have been removed "
            "for bucket calculation" % num_nones
        )

    print(
        'Collected %i control vals between %.6f and %.6f'
        % (len(ctrl_vals), min(ctrl_vals), max(ctrl_vals))
    )

    # Calculate bucket lower bounds
    print('Calculating lowerbounds for %i buckets...' % num_buckets)
    ctrl_vals = sorted(ctrl_vals)
    lb_indices = [int(len(ctrl_vals) * i / num_buckets) for i in range(num_buckets)]
    lbs = [ctrl_vals[idx] for idx in lb_indices]
    print('\nBucket lowerbounds for control %s: ' % ctrl)
    print(lbs)

    # Calculate the actual bucket sizes
    bucket_sizes = Counter()
    bucket_ids = [sort_into_bucket(ctrl_val, lbs) for ctrl_val in ctrl_vals]
    bucket_sizes.update(bucket_ids)
    print('\nBucket sizes: ')
    for bucket_id in sorted(bucket_sizes.keys()):
        print("%i: %i" % (bucket_id, bucket_sizes[bucket_id]))


def main():
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
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument(
        '--control',
        type=str,
        default='',
        help='the control for which we want to calculate the buckets',
    )
    parser.add_argument(
        '--num-buckets',
        type=int,
        default=10,
        help='the number of buckets we want to calculate',
    )

    parser.set_defaults(task="projects.controllable_dialogue.tasks.agents")
    parser.set_defaults(datatype="train:stream")

    opt = parser.parse_args()

    bucket_data(opt)


if __name__ == '__main__':
    main()
