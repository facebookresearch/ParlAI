#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from .build import build

import json
import os


class DefaultTeacher(DialogTeacher):
    """
    MutualFriends dataset.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)

        if not opt['datatype'].startswith('train'):
            raise RuntimeError('MutualFriends only has a training set.')
        opt['datafile'] = os.path.join(opt['datapath'], 'MutualFriends', 'data.json')
        self.id = 'mutualfriends'
        super().__init__(opt, shared)

    def act(self):
        """
        Use DialogTeacher act but set id to "Teacher" for intro message.
        """
        reply = super().act()
        if reply.get('text', '').startswith('You have the following friends'):
            reply['id'] = 'Teacher'
        return reply

    def setup_data(self, path):
        """
        Load json data of conversations.
        """
        print('loading: ' + path)
        with open(path) as data_file:
            self.loaded_data = json.load(data_file)
        for ex in self.loaded_data:
            if len(ex['events']) > 0:
                # TODO: add reverse conversation as well
                curr_agent = ex['events'][0]['agent']
                conversation = [
                    (
                        'You have the following friends:\n'
                        + '\n'.join(
                            ', '.join('{}={}'.format(k, v) for k, v in person.items())
                            for person in ex['scenario']['kbs'][int(curr_agent)]
                        )
                        + '\nTry to find out which friend the other person has in common.'
                    )
                ]
                curr = ''
                idx = 0
                while idx < len(ex['events']):
                    msg = ex['events'][idx]['data']
                    if type(msg) == dict:
                        msg = 'SELECT({})'.format(
                            ', '.join('{}={}'.format(k, v) for k, v in msg.items())
                        )
                    next_agent = ex['events'][idx]['agent']
                    if curr_agent == next_agent:
                        curr += '\n' + msg
                        curr = curr.strip()
                    else:
                        conversation.append(curr)
                        curr = msg
                        curr_agent = next_agent
                    idx += 1
                conversation.append(curr)
                for i in range(0, len(conversation), 2):
                    if i + 1 < len(conversation) - 1:
                        yield (conversation[i], [conversation[i + 1]]), i == 0
                    elif i + 1 == len(conversation) - 1:
                        yield (
                            (conversation[i], [conversation[i + 1]], ex['outcome']),
                            False,
                        )
                    else:
                        yield (conversation[i], None, ex['outcome']), False
