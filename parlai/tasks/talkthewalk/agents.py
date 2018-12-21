#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FixedDialogTeacher
from .build import build
from .worlds import Simulator, is_action
import json
import os


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    opt['ttw_data'] = os.path.join(opt['datapath'], 'TalkTheWalk')
    return opt['ttw_data'], os.path.join(opt['ttw_data'],
                                         'talkthewalk.' + dt + '.json')


class TTWTeacher(FixedDialogTeacher):

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Talk the Walk Teacher Arguments')
        agent.add_argument('--train-actions',
                           type='bool', default=False, help='Train model to \
                           take actions')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        data_path, datafile = _path(opt)
        self.label_candidates = set()

        if shared:
            self.data = shared['data']
            self.sim = shared['sim']
            self.label_candidates = shared['cands']
        else:
            self.sim = Simulator(opt)
            self._setup_data(datafile)
        self.reset()

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['sim'] = self.sim
        shared['cands'] = self.label_candidates
        return shared


    def _setup_episode(episode):
        """Process one episode in an example."""
        raise NotImplementedError(
            'Abstract class: user must implement _setup_episode')

    def _setup_data(self, datafile):
        self.episodes = json.load(open(datafile))
        self.data = []
        self.examples_count = 0

        for episode in self.episodes:
            if episode:
                init = {x: y for x, y in episode.items() if x in ['start_location',
                    'neighborhood', 'boundaries', 'target_location']}
                self.sim.init_sim(**init)

                episode = self._setup_episode(episode)

                if episode:
                    self.label_candidates = self.label_candidates.union(
                            [x['labels'][0] for x in episode])
                    self.data.append(episode)
                    self.examples_count += len(episode)

    def get(self, episode_idx, entry_idx=0):
        example = self.data[episode_idx][entry_idx]
        example['text'] = example.get('text', '__silence__')
        example['label_candidates'] = self.label_candidates
        return example

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return self.examples_count


class TouristTeacher(TTWTeacher):
    def _setup_episode(self, episode):
        ep = []
        example = {'episode_done': False}
        for msg in episode['dialog']:
            text = msg['text']
            if msg['id'] == 'Tourist':
                if self.opt.get('train_actions') or not is_action(text):
                    example['labels'] = [text]
                    ep.append(example)
                    example = {'episode_done': False}
                # add movements to text history if not training on them
                if not self.opt.get('train_actions') and is_action(text):
                    example['text'] = example.get('text', '') + text + '\n'
            elif msg['id'] == 'Guide':
                example['text'] = example.get('text', '') + text + '\n'

            self.sim.execute(text)
            self.sim.add_view_to_text(example, text)

        if len(ep):
            ep[-1]['episode_done'] = True
        return ep


class GuideTeacher(TTWTeacher):
    def _setup_episode(self, episode):
        ep = []
        example = {'episode_done': False, 'text': self.sim.get_text_map()}
        for msg in episode['dialog']:
            text = msg['text']
            if msg['id'] == 'Guide':
                if self.opt.get('train_actions') or not text.startswith('EVALUATE'):
                    example['labels'] = [text]
                    ep.append(example)
                    example = {'episode_done': False}
            elif msg['id'] == 'Tourist' and not is_action(text):
                example['text'] = example.get('text', '') + text + '\n'

            self.sim.execute(text)

        if len(ep):
            ep[-1]['episode_done'] = True
        return ep


class GuideLocalizeTeacher(TTWTeacher):
    def _setup_episode(self, episode):
        ep = []
        example = {'episode_done': False, 'text': self.sim.get_text_map()}
        for msg in episode['dialog']:
            text = msg['text']
            if msg['id'] == 'Guide':
                example['text'] = example.get('text', '') + text + '\n'
            elif msg['id'] == 'Tourist' and not is_action(text):
                example['text'] = example.get('text', '') + text + '\n'
                example['labels'] = [self.sim.get_agent_location()]
                ep.append(example)
                example = {'episode_done': False}

            self.sim.execute(text)

        if len(ep):
            ep[-1]['episode_done'] = True
        return ep


class DefaultTeacher(TouristTeacher):
    pass
