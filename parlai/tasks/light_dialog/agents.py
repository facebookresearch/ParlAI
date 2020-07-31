#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import ParlAIDialogTeacher
from .build import build

import copy
import os
import random


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    # for now train, valid and test will be identical, will change with more data.
    dt = opt['datatype'].split(':')[0]
    if (
        opt.get('light_unseen_test', False) is True
        or opt.get('light_unseen_test', False) == 'True'
    ):
        if dt == 'test':
            dt = 'test_unseen'
        else:
            raise ValueError('No unseen train or valid.')
    fields = [
        'taskname',
        'setting',
        'objects',
        'person_names',
        'persona',
        'emote',
        'speech',
        'action',
        'affordances',
        'repeat',
        'cands',
        'current_self_output',
        'clip_cands',
        'speech_prefix',
    ]
    fpath = ''
    for f in fields:
        fpath += f + str(opt['light_use_' + f]) + "_"
    return os.path.join(
        opt['datapath'],
        'light_dialogue',
        fpath[:-1],
        opt['light_label_type'] + '_' + dt + '.txt',
    )


class DefaultTeacher(ParlAIDialogTeacher):
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('LIGHT Dialogue options')
        agent.add_argument(
            '--light_use_repeat',
            type=str,
            default='none',
            choices=['self_last', 'partner_last', 'none', 'both_last'],
        )
        agent.add_argument('--light_use_taskname', type='bool', default=True)
        agent.add_argument('--light_use_setting', type='bool', default=True)
        agent.add_argument('--light_unseen_test', type='bool', default=False)
        agent.add_argument('--light_use_person_names', type='bool', default=True)
        agent.add_argument(
            '--light_use_persona',
            type=str,
            default='self',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument('--light_use_objects', type='bool', default=True)
        agent.add_argument(
            '--light_use_emote',
            type=str,
            default='all',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument(
            '--light_use_speech',
            type=str,
            default='all',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument(
            '--light_use_action',
            type=str,
            default='all',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument('--light_use_affordances', type='bool', default=True)
        agent.add_argument(
            '--light_use_current_self_output',
            type=str,
            default="all",
            choices=['none', 'all', 'all_filtered', 'all_filtered_remove'],
        )
        agent.add_argument(
            '--light_label_type',
            type=str,
            default='speech',
            choices=['speech', 'action', 'emote'],
            help='type of target in light dialogues',
        )
        agent.add_argument('--light_use_cands', type=int, default=20)
        agent.add_argument('--light_use_clip_cands', type=int, default=10000)
        agent.add_argument('--light_speech_prefix', type='bool', default=True)
        agent.add_argument(
            '--light_percent_train_exs',
            type=float,
            default=1.0,
            help='Float in range [0, 1] indicating proportion of train set to use',
        )

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        self.pct_train_exs = opt['light_percent_train_exs']
        assert 0.0 <= self.pct_train_exs <= 1.0
        opt['parlaidialogteacher_datafile'] = _path(opt)
        if 'light_use_speech_prefix' not in opt:
            opt['light_use_speech_prefix'] = True
        super().__init__(opt, shared)

    def _setup_data(self, path):
        """
        Overriding to limit num train exs.
        """
        super()._setup_data(path)
        if self.training and self.pct_train_exs <= 1.0:
            random.seed(42)
            self.episodes = random.sample(
                self.episodes, int(self.num_episodes() * self.pct_train_exs)
            )
            self.num_exs = sum(len(e) for e in self.episodes)


class SimpleTeacher(DefaultTeacher):
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('LIGHT Dialogue options')
        agent.add_argument(
            '--light_use_repeat',
            type=str,
            default='none',
            choices=['self_last', 'partner_last', 'none', 'both_last'],
        )
        agent.add_argument('--light_use_taskname', type='bool', default=True)
        agent.add_argument('--light_use_setting', type='bool', default=True)
        agent.add_argument('--light_unseen_test', type='bool', default=False)
        agent.add_argument('--light_use_person_names', type='bool', default=True)
        agent.add_argument(
            '--light_use_persona',
            type=str,
            default='self',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument('--light_use_objects', type='bool', default=False)
        agent.add_argument(
            '--light_use_emote',
            type=str,
            default='none',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument(
            '--light_use_speech',
            type=str,
            default='partner',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument(
            '--light_use_action',
            type=str,
            default='none',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument('--light_use_affordances', type='bool', default=False)
        agent.add_argument(
            '--light_use_current_self_output',
            type=str,
            default="none",
            choices=['none', 'all', 'all_filtered', 'all_filtered_remove'],
        )
        agent.add_argument(
            '--light_label_type',
            type=str,
            default='speech',
            choices=['speech', 'action', 'emote'],
            help='type of target in light dialogues',
        )
        agent.add_argument('--light_use_cands', type=int, default=20)
        agent.add_argument('--light_use_clip_cands', type=int, default=10000)
        agent.add_argument('--light_use_speech_prefix', type='bool', default=False)
        agent.add_argument('--light_percent_train_exs', type=float, default=1.0)

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id += '_' + self.opt['light_label_type']


class SimpleMultiTeacher(DefaultTeacher):
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('LIGHT Dialogue options')
        agent.add_argument(
            '--light_use_repeat',
            type=str,
            default='none',
            choices=['self_last', 'partner_last', 'none', 'both_last'],
        )
        agent.add_argument('--light_use_setting', type='bool', default=True)
        agent.add_argument('--light_unseen_test', type='bool', default=False)
        agent.add_argument('--light_use_person_names', type='bool', default=True)
        agent.add_argument(
            '--light_use_persona',
            type=str,
            default='self',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument('--light_use_taskname', type='bool', default=False)
        agent.add_argument('--light_use_objects', type='bool', default=False)
        agent.add_argument(
            '--light_use_emote',
            type=str,
            default='none',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument(
            '--light_use_speech',
            type=str,
            default='partner',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument(
            '--light_use_action',
            type=str,
            default='none',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument('--light_use_affordances', type='bool', default=False)
        agent.add_argument(
            '--light_use_current_self_output',
            type=str,
            default="all",
            choices=['none', 'all', 'all_filtered', 'all_filtered_remove'],
        )
        agent.add_argument(
            '--light_label_type',
            type=str,
            default='speech',
            choices=['speech', 'action', 'emote', 'which'],
            help='type of target in light dialogues',
        )
        agent.add_argument('--light_use_cands', type=int, default=20)
        agent.add_argument('--light_use_clip_cands', type=int, default=10000)
        agent.add_argument('--light_use_speech_prefix', type='bool', default=False)
        agent.add_argument('--light_percent_train_exs', type=float, default=1.0)

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id += '_' + self.opt['light_label_type']


class SelfchatTeacher(SimpleTeacher):
    """
    Teacher used to create candidates for selfchats, if needed.
    """

    pass
