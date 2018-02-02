# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Builds ParlAI data into the Tensorflow Seq2Seq parallel data format."""

import copy
import os
import random
import tempfile

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from build_dict import build_dict


class ToFileAgent(Agent):
    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument(
            '--output_dir', default=None, help='Directory to write files to.')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.output_dir = opt['output_dir']
        print('Saving files to directory {}'.format(self.output_dir))

        self.dt = opt['datatype'].split(':')[0]
        xfn = os.path.join(self.output_dir, '_'.join([opt['task'], self.dt, 'texts.txt']))
        yfn = os.path.join(self.output_dir, '_'.join([opt['task'], self.dt, 'labels.txt']))
        self.xs = open(xfn, 'w')
        self.ys = open(yfn, 'w')
        self.cnt = 0

    def strip(self, text):
        """Remove newlines, since writing to newline-sep file."""
        return text.replace('\n', ' ').strip()

    def act(self):
        obs = self.observation
        if 'text' in obs:
            labels = obs.get('labels', obs.get('eval_labels'))
            if labels is not None:
                self.xs.write(self.strip(obs['text']) + '\n')
                self.ys.write(self.strip(random.choice(labels)) + '\n')
                self.cnt += 1
                return {'text': 'Example written.'}
        return {}

    def shutdown(self):
        self.xs.close()
        self.ys.close()
        print('[ ToFileAgent: {} {} examples written to {}/ ]'.format(
            self.cnt, self.dt, self.output_dir))


def setup_args():
    parser = ParlaiParser()
    parser.add_argument('--output_dir', default=None,
        help='Directory to write files to.')
    DictionaryAgent.add_cmdline_args(parser)
    return parser

def build_parallel_data(parser):
    # override opts
    opt = parser.parse_args()
    if opt.get('output_dir') is None:
        opt['output_dir'] = tempfile.mkdtemp()
    opt['image_mode'] = 'none'
    for datatype in 'train', 'valid', 'test':
        opt['datatype'] = datatype + ':ordered:stream'
        writer = ToFileAgent(opt)
        world = create_task(opt, writer)
        while not world.epoch_done():
            world.parley()
        world.shutdown()
    opt['dict_file'] = os.path.join(opt['output_dir'], opt['task'] + '_dict.tsv')
    build_dict(opt, freqs=False)
    return opt['output_dir']


if __name__ == '__main__':
    build_parallel_data(setup_args())
