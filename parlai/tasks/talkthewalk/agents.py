#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import TTWBase
from .worlds import is_action


class TouristTeacher(TTWBase):
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


class GuideTeacher(TTWBase):
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


class GuideLocalizeTeacher(TTWBase):
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
