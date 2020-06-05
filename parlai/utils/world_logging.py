#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Useful utilities for logging actions/observations in a world.
"""

from parlai.core.worlds import BatchWorld
from parlai.utils.misc import msg_to_str
from parlai.utils.conversations import Conversations
import parlai.utils.logging as logging

import copy
from tqdm import tqdm

KEEP_ALL = 'all'


class WorldLogger:
    """
    Logs actions/observations in a world and saves in a given format.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('World Logging')
        agent.add_argument(
            '--log-keep-fields',
            type=str,
            default=KEEP_ALL,
            help='Fields to keep when logging. Should be a comma separated list',
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
        parleys = zip(*batch_act)
        for i, parley in enumerate(parleys):
            self._add_msgs(parley, idx=i)
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

    def convert_to_labeled_data(self, episode):
        out = []
        text_lst = []
        for parley in episode:
            first_act, second_act = parley
            if 'text' in first_act:
                text_lst.append(first_act['text'])
            if second_act.get('id') != 'context':
                label = second_act.get('text')
                out.append(
                    {
                        'id': first_act.get('id', ''),
                        'text': '\n'.join(text_lst),
                        'labels': [label],
                        'episode_done': False,
                    }
                )
                text_lst = []
        if len(out) > 0:
            out[-1]['episode_done'] = True
        return out

    def write_parlai_format(self, outfile):
        logging.info(f'Saving log to {outfile} in ParlAI format')
        with open(outfile, 'w') as fw:
            for episode in tqdm(self._logs):
                ep = self.convert_to_labeled_data(episode)
                for act in ep:
                    txt = msg_to_str(act)
                    fw.write(txt + '\n')
                fw.write('\n')

    def write_conversations_format(self, outfile, world):
        Conversations.save_conversations(
            self._logs,
            outfile,
            world.opt,
            self_chat=world.opt.get('selfchat_task', False),
        )

    def write(self, outfile, world, file_format='conversations', indent=4):
        if file_format == 'conversations':
            self.write_conversations_format(outfile, world)
        else:
            # ParlAI text format
            self.write_parlai_format(outfile)

    def get_logs(self):
        return self._logs
