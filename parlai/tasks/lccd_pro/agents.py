#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This task simply loads the specified file: useful for quick tests without
# setting up a new task.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.utils.misc import warn_once
import parlai.utils.logging as logging
from parlai.utils.io import PathManager
from parlai.utils.conversations import Conversation
import json
import copy
import os
from .get_data import build

from parlai.core.teachers import ConversationTeacher

def _path(opt, filtered):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    #datatype = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'LCCD_pro', 'LCCD_pro' + '.json')
    

class LCCDTeacher(ConversationTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('LCCD Task Arguments')
        agent.add_argument(
            '--label-turns',
            type=str,
            help='which speaker to use as label',
            choices=['firstspeaker', 'secondspeaker', 'both'],
            default='secondspeaker',
        )
        return parser

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        # get datafile
        opt['conversationteacher_datafile'] = _path(opt, '')
        super().__init__(opt, shared)

    def setup_data(self, path):
        logging.info(f"[loading data from json file into task: {path} ]")
        # conversations = Conversations(path)  # slow down the speed when json file is big.
        with PathManager.open(path, 'r') as f:
            # lines = f.read().splitlines()
            for line in f:
                conv = Conversation(json.loads(line))
                if conv.context:
                    warn_once(
                    'At least one of these conversations contains a context, which is not being used'
                    )
                turns = [t for t in conv.turns if t.get('id') != 'context']
                if len(turns) != len(conv.turns):
                    warn_once(
                    'At least one of these conversations contains a context within the dialogue, which is being discarded'
                    )
                turns.insert(0, Message({'text': '__SILENCE__'}))
                # train on odd turns as labels (turns w/ first speaker)
                if self.label_turns in ['firstspeaker', 'both']:
                    eps = self._get_ep_from_turns(turns[::2], turns[1::2])
                    if eps:
                        for example, example_begins in self._return_episode_examples(eps):
                            yield example, example_begins
                # train on even turns as labels (turns w/ second speaker)
                if self.label_turns in ['secondspeaker', 'both']:
                    eps = self._get_ep_from_turns(turns[1::2], turns[2::2])
                    if eps:
                        for example, example_begins in self._return_episode_examples(eps):
                            yield example, example_begins

class DefaultTeacher(LCCDTeacher):
    pass