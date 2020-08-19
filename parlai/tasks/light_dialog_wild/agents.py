#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import ParlAIDialogTeacher
from .build import build, get_fpath

import copy
import os
import random


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    # for now train, valid and test will be identical, will change with more data.
    dt = opt['datatype'].split(':')[0]
    fpath = get_fpath(opt)

    return os.path.join(
        opt['datapath'],
        'light_dialogue_wild',
        fpath,
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
        agent.add_argument('--light_use_unseen_test', type='bool', default=False)
        agent.add_argument('--light_use_person_names', type='bool', default=True)
        agent.add_argument(
            '--light_use_persona',
            type=str,
            default='self',
            choices=['partner', 'self', 'all', 'none'],
        )
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
        agent.add_argument('--light_use_speech_prefix', type='bool', default=True)
        agent.add_argument('--light_use_score_cutoff', type=int, default=1)
        agent.add_argument('--light_use_max_score_cutoff', type=int, default=-1)
        agent.add_argument(
            '--light_use_hard_score_cutoff',
            type='bool',
            default=False,
            help='Specify True to **only** include examples with the specified '
            'score cutoff. E.g., if `--light-use-score-cutoff 3`, will only '
            'supply dialogues with scores of 3.',
        )
        agent.add_argument(
            '--light-model-name',
            type=str,
            default=None,
            help='if specified, the model from which chats should be saved '
            'to specify multiple, delimit with + symbol',
        )
        agent.add_argument(
            '--light-use-continue-type',
            type=str,
            default='all',
            choices=['all', 'continue', 'exit'],
            help='only use dialogues that had a specific continue outcome, default use all',
        )
        agent.add_argument(
            '--light-use-date-cutoff',
            type=str,
            default=None,
            help=(
                'If specified, only include hobbot conversations that were collected '
                'before this date. Format YYYY-MM-DD.\n'
                'First wild experiments ended 2020-05-26.\n'
                'First model group ended 2020-06-23.\n'
            ),
        )
        agent.add_argument(
            '--light-use-person-names-prefix',
            type='bool',
            default=False,
            help="If specified, prefix text with character name",
        )
        agent.add_argument('--light_percent_train_exs', type=float, default=1.0)

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _path(opt)
        if 'light_use_speech_prefix' not in opt:
            opt['light_use_speech_prefix'] = True
        self.pct_train_exs = opt['light_percent_train_exs']
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
        agent.add_argument('--light_use_unseen_test', type='bool', default=False)
        agent.add_argument('--light_use_person_names', type='bool', default=True)
        agent.add_argument(
            '--light_use_persona',
            type=str,
            default='self',
            choices=['partner', 'self', 'all', 'none'],
        )
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
        agent.add_argument('--light_use_score_cutoff', type=int, default=1)
        agent.add_argument('--light_use_max_score_cutoff', type=int, default=-1)
        agent.add_argument(
            '--light_use_hard_score_cutoff',
            type='bool',
            default=False,
            help='Specify True to **only** include examples with the specified '
            'score cutoff. E.g., if `--light-use-score-cutoff 3`, will only '
            'supply dialogues with scores of 3.',
        )
        agent.add_argument(
            '--light-model-name',
            type=str,
            default=None,
            help='if specified, the model from which chats should be saved',
        )
        agent.add_argument(
            '--light-use-continue-type',
            type=str,
            default='all',
            choices=['all', 'continue', 'exit'],
            help='only use dialogues that had a specific continue outcome, default use all',
        )
        agent.add_argument(
            '--light-use-date-cutoff',
            type=str,
            default=None,
            help=(
                'If specified, only include hobbot conversations that were collected '
                'before this date. Format YYYY-MM-DD.'
            ),
        )
        agent.add_argument(
            '--light-use-person-names-prefix',
            type='bool',
            default=False,
            help="If specified, prefix text with character name",
        )
        agent.add_argument('--light_percent_train_exs', type=float, default=1.0)

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id += '_' + self.opt['light_label_type']


class SimpleMultiTeacher(DefaultTeacher):
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('LIGHT Dialogue options')
        agent.add_argument(
            '--light_use_repeat',
            type=str,
            default='none',
            choices=['self_last', 'partner_last', 'none', 'both_last'],
        )
        agent.add_argument('--light_use_setting', type='bool', default=True)
        agent.add_argument('--light_use_unseen_test', type='bool', default=False)
        agent.add_argument('--light_use_person_names', type='bool', default=True)
        agent.add_argument(
            '--light_use_persona',
            type=str,
            default='self',
            choices=['partner', 'self', 'all', 'none'],
        )
        agent.add_argument('--light_use_taskname', type='bool', default=False)
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
        agent.add_argument('--light_use_score_cutoff', type=int, default=1)
        agent.add_argument('--light_use_max_score_cutoff', type=int, default=-1)
        agent.add_argument(
            '--light_use_hard_score_cutoff',
            type='bool',
            default=False,
            help='Specify True to **only** include examples with the specified '
            'score cutoff. E.g., if `--light-use-score-cutoff 3`, will only '
            'supply dialogues with scores of 3.',
        )
        agent.add_argument(
            '--light-model-name',
            type=str,
            default=None,
            help='if specified, the model from which chats should be saved',
        )
        agent.add_argument(
            '--light-use-continue-type',
            type=str,
            default='all',
            choices=['all', 'continue', 'exit'],
            help='only use dialogues that had a specific continue outcome, default use all',
        )
        agent.add_argument(
            '--light-use-date-cutoff',
            type=str,
            default=None,
            help=(
                'If specified, only include hobbot conversations that were collected '
                'before this date. Format YYYY-MM-DD.'
            ),
        )
        agent.add_argument(
            '--light-use-person-names-prefix',
            type='bool',
            default=False,
            help="If specified, prefix text with character name",
        )
        agent.add_argument('--light_percent_train_exs', type=float, default=1.0)

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id += '_' + self.opt['light_label_type']


class ReversedTeacher(DefaultTeacher):
    """
    Reversed teacher where labels are what bot said, text is what human said.

    Only support swapping text & labels, no other guarantees.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        DefaultTeacher.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('LIGHT Dialogue options')
        agent.add_argument(
            '--light-min-conv-length',
            type=int,
            default=None,
            help="if specified, filters out conversations that do not reach this length. "
            "`--light-min-conv-length 3` will filter out convos with less than 3 turns",
        )
        agent.add_argument(
            '--light-max-conv-length',
            type=int,
            default=None,
            help="if specified, caps conversations at this length. "
            "`--light-max-conv-length 6` will cap all conversations at 6 turns",
        )
        agent.add_argument(
            '--light-hard-max-length-cap',
            type='bool',
            default=False,
            help="if specificed, conversation max length is a hard cap - that is, "
            "any conversations greater than --light-max-conv-length will be "
            "removed completely",
        )
        agent.add_argument(
            '--light-prettify-setting',
            type='bool',
            default=False,
            help='if specified, setting/characters prettified.',
        )

    def _setup_data(self, path):
        super()._setup_data(path)
        new_eps = []
        min_len = self.opt.get('light_min_conv_length')
        if min_len:
            self.episodes = [e for e in self.episodes if len(e) >= min_len + 1]
        for ep in self.episodes:
            texts = [ex['text'] for ex in ep]
            labels = [ex['labels'] for ex in ep]
            new_ep = []
            context = texts[0]
            if self.opt.get('light_use_setting'):
                if self.opt.get('light_prettify_setting') and self.opt.get(
                    'light_use_person_names_prefix'
                ):
                    char1, char2 = [texts[1].split(':')[0], labels[0][0].split(':')[0]]
                    name, desc = context.split('_setting_desc')
                    name = name.replace('_setting_name', '').replace(', Somewhere', '')
                    context = (
                        f"A {char1} and a {char2} are in {name}. {desc.split('.')[0]}"
                    )
                new_ep = [
                    {'text': '', 'labels': [f"*{context}*"], 'episode_done': False}
                ]
            new_ep += [
                {'text': labels[i][0], 'labels': [texts[i + 1]], 'episode_done': False}
                for i in range(0, len(texts) - 1)
            ]
            if new_ep:
                new_ep[-1]['episode_done'] = True
                new_eps.append(new_ep)

        max_len = self.opt.get('light_max_conv_length')
        hard_max_cap = self.opt.get('light_hard_max_length_cap')
        if max_len:
            if hard_max_cap:
                new_eps = [e for e in new_eps if len(e) <= max_len]
            else:
                new_eps = [e[:max_len] for e in new_eps]
                for e in new_eps:
                    e[-1]['episode_done'] = True

        self.episodes = new_eps
        self.num_exs = sum([len(ep) for ep in self.episodes])


class SelfchatTeacher(SimpleTeacher):
    """
    Teacher used to create candidates for selfchats, if needed.
    """

    pass
