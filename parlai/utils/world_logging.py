#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Useful utilities for logging actions/observations in a world.
"""

from parlai.utils.misc import msg_to_str
import copy
import json


class WorldLogger:
    """
    Logs actions/observations in a world.
    """

    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)
        self.keep_fields = opt.get('log_keep_fields', 'id,text,episode_done').split(',')
        self.reset()

    def reset(self):
        self._logs = []
        self._log = []
        self.add_episode(self._log)

    def add_msgs(self, acts):
        msgs = []
        for act in acts:
            m = {}
            for f in self.keep_fields:
                if f in act:
                    m[f] = act[f]
            msgs.append(m)
        self._log.append(msgs)

    def add_episode(self, log):
        self._logs.append(log)

    def log(self, world):
        acts = world.get_acts()
        self.add_msgs(acts)
        if world.episode_done():
            # add episode to logs
            self._log = []
            self.add_episode(self._log)

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
        print('[ saving log to {} ]'.format(outfile))
        fw = open(outfile, 'w')
        for episode in self._logs:
            ep = self.convert_to_labeled_data(episode)
            for a in ep:
                txt = msg_to_str(a)
                fw.write(txt + '\n')
            fw.write('\n')
        fw.close()

    def write_json_format(self, outfile):
        print('[ saving log to {} ]'.format(outfile))
        with open(outfile, 'w') as of:
            json.dump(self._logs, of)

    def write(self, outfile, file_format):
        if file_format == 'json':
            self.write_json_format(outfile)
        else:
            self.write_parlai_format(outfile)
