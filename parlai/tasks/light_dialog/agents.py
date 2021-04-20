#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
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
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('LIGHT Dialogue options')
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
        return parser

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
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('LIGHT Dialogue options')
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
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id += '_' + self.opt['light_label_type']


class SimpleMultiTeacher(DefaultTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('LIGHT Dialogue options')
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
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id += '_' + self.opt['light_label_type']


class SelfchatTeacher(SimpleTeacher):
    """
    Teacher used to create candidates for selfchats, if needed.
    """

    pass


class ContextGenerator:
    """
    Generates contexts shown to crowdsourced workers when collecting LIGHT conversations.

    This generator was used to generate the context information shown to workers at the
    beginning of a conversation/.
    """

    SETTING_NAME = '_setting_name '
    SETTING_DESC = '_setting_desc '
    SELF_NAME = '_self_name '
    SELF_PERSONA = '_self_persona '
    PARTNER_NAME = '_partner_name '
    SELF_SAY = '_self_say '
    PARTNER_SAY = '_partner_say '

    def __init__(self, opt, datatype: str = 'test', seed: Optional[int] = None):
        """
        Initalize the context generator.

        opt: only a 'datapath' key is required, to specify the ParlAI data folder
        """
        import json

        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

        with open(opt['persona_path']) as f:
            self.personas = json.load(f)

    def get_context(self) -> dict:
        """
        Get context information to be shown at the beginning of one conversation.

        Values in return dict:
        - context_dataset: the dataset (ConvAI2, EmpatheticDialogues, or Wizard of
            Wikipedia) used to generate the context information.
        - persona_1_strings, persona_2_strings: 2 persona strings each for the two
            speakers, chosen randomly from the ConvAI2 dataset. If context_dataset ==
            "wizard_of_wikipedia", these persona strings will be matched to the WoW
            topic returned in the "additional_context" field.
        - additional_context: provides additional bits of information to give context
            for the speakers. If context_dataset == "empathetic_dialogues", this is a
            situation from the start of an ED conversation. If context_dataset ==
            "wizard_of_wikipedia", this is a topic from the WoW dataset that matches the
            persona strings. If context_dataset == "convai2", this is None.
        - person1_seed_utterance, person2_seed_utterance: two lines of a conversation
            from the dataset specified by "context_dataset". They will be shown to the
            speakers to "seed" the conversation, and the speakers continue from where
            the lines left off.
        """
        personas = random.sample(self.personas, 2)
        (human_name, human_persona_text, loc1), (
            bot_name,
            bot_persona_text,
            loc2,
        ) = personas
        location = random.choice([loc1, loc2])
        loc_name, loc_desc = location.split(', ', 1)

        bot_persona_msg = '\n'.join(
            [
                "_task_speech",
                self.SETTING_NAME + loc_name,
                self.SETTING_DESC + loc_desc,
                self.PARTNER_NAME + human_name,
                self.SELF_NAME + bot_name,
                self.SELF_PERSONA + bot_persona_text,
            ]
        )

        human_persona_msg = '\n'.join(
            [
                f"You'll be playing the role of: '{human_name}'. Your persona is: '{human_persona_text}'.",
                f"You are in the '{loc_name}: {loc_desc}' Your chat partner is '{bot_name}'",
            ]
        )

        return {
            'context_dataset': 'light',
            'persona_1_strings': bot_persona_msg,
            'persona_2_strings': human_persona_msg,
        }
