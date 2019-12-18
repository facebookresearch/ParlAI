#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Useful utilities for logging actions/observations in a world.
"""

from parlai.core.worlds import BatchWorld
from parlai.utils.misc import msg_to_str

import copy
import json
from tqdm import tqdm

KEEP_ALL = 'all'


class WorldLogger:
    """
    Logs actions/observations in a world and saves in a JSONL format.
    """
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('World Logging')
        agent.add_argument(
            '--log-keep-fields',
            type=str,
            default=KEEP_ALL,
            help='Fields to keep when logging. Should be a comma separated list'
        )

    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)
        self._set_keep_fields(opt)

        self._current_episodes = {}
        self._logs = []

        self.reset()

    def _set_keep_fields(self, opt):
        self.keep_fields = opt['log_keep_fields'].split(',')
        self.keep_all = KEEP_ALL in self.keep_fields

    def reset(self):
        for _, ep in self._current_episodes.items():
            self._add_episode(ep)

        self._current_episodes = {}

    def reset_world(self, idx=0):
        self._add_episode(self._current_episodes[idx])
        self._current_episodes[idx] = []

    def _add_msgs(self, acts, idx=0):
        """
        Add messages from a `parley()` to the current episode of logs.

        :param acts: list of acts from a `.parley()` call
        """
        msgs = []
        for act in acts:
            if not self.keep_all:
                msg = {f: act[f] for f in self.keep_fields if f in act}
            else:
                msg = act
            msgs.append(msg)

        self._current_episodes.setdefault(idx, [])
        self._current_episodes[idx].append(msgs)

    def _add_episode(self, episode):
        """
        Add episode to the logs.
        """
        self._logs.append(episode)

    def _is_batch_world(self, world):
        return isinstance(world, BatchWorld) and len(world.worlds) > 1

    def _log_batch(self, world):
        batch_act = world.get_acts()
        act_pairs = zip(*batch_act)
        for i, act_pair in enumerate(act_pairs):
            self._add_msgs(act_pair, idx=i)
            if world.worlds[i].episode_done():
                self.reset_world(idx=i)

    def log(self, world):
        """
        Log acts from a world.
        """
        # log batch world
        if self._is_batch_world(world):
            self._log_batch(world)
            return

        # log single world
        acts = world.get_acts()
        self._add_msgs(acts)
        if world.episode_done():
            # add episode to logs and clear examples
            self.reset_world()

    def convert_to_labeled_data(self, log):
        out = []
        text = ''
        for msgs in log:
            if text != '':
                text += '\n'
            text += msgs[0].get('text')
            if msgs[1].get('id') != 'context':
                label = msgs[1].get('text')
                out.append(
                    {
                        'id': msgs[0].get('id'),
                        'text': text,
                        'labels': [label],
                        'episode_done': False,
                    }
                )
                text = ''
        if len(out) > 0:
            out[-1]['episode_done'] = True
        return out

    def write_parlai_format(self, outfile):
        print('[ Saving log to {} in ParlAI format ]'.format(outfile))
        with open(outfile, 'w') as fw:
            for episode in tqdm(self._log):
                ep = self.convert_to_labeled_data(episode)
                for a in ep:
                    txt = msg_to_str(a)
                    fw.write(txt + '\n')
                fw.write('\n')

    def write_jsonl_format(self, outfile):
        print('[ Saving log to {} in jsonl format ]'.format(outfile))
        with open(outfile, 'w') as of:
            for episode in tqdm(self._logs):
                dialog = {'dialog': episode}
                json_episode = json.dumps(dialog)
                of.write(json_episode + '\n')

    def write(self, outfile, file_format='jsonl'):
        if file_format == 'jsonl':
            self.write_json_format(outfile)
        else:
            self.write_parlai_format(outfile)
