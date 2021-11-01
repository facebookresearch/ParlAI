#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.params import ParlaiParser
from parlai.agents.fixed_response.fixed_response import FixedResponseAgent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
import jsonlines
from tqdm import tqdm

import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Task convs concat')
    parser.add_argument('--export-fpath', type=str, default='./exported_file.json')
    return parser


def example_strs(opt, world):
    out_strs = []
    if opt['batchsize'] > 1:
        raise RuntimeError('Simple view only support batchsize=1')
    act = world.get_acts()[0]
    text = act.get('text', '[no text field]')
    out_strs.append(text)
    labels = act.get('labels', act.get('eval_labels', ['[no labels field]']))
    labels = '|'.join(labels)
    out_strs.append(labels)
    return out_strs


def concat_conv_data(opt):

    opt['fixed_response'] = None
    agent = FixedResponseAgent(opt)
    world = create_task(opt, agent)

    all_examples = []
    this_episode = []
    for _ in tqdm(range(world.num_examples())):
        world.parley()
        this_episode.extend(example_strs(opt, world))
        if world.get_acts()[0]['episode_done']:
            all_examples.append(
                {'text': '\n'.join(this_episode), 'task_name': opt['task']}
            )
            this_episode = []
        if world.epoch_done():
            break

    with jsonlines.open(opt['export_fpath'], 'w') as fout:
        for episode in all_examples:
            fout.write(episode)


class ConcatData(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return concat_conv_data(self.opt)


if __name__ == '__main__':
    ConcatData.main()
