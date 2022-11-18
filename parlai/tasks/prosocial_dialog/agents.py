#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Optional

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import DialogTeacher

from .build import build


class ProsocialDialogSafetyTeacher(DialogTeacher):
    """
    Safety Teacher for ProsocialDialog Data https://github.com/skywalker023/prosocial-dialog
    set --one-turn to true for just one turn without the context.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('prosocial dialog safety options')
        agent.add_argument(
            '--one-turn',
            type=bool,
            default=False,
            help="Whether or not to have the text include the context if it exists or just single turn",
        )
        return parser

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        self.datatype = opt['datatype'].split(':')[0]
        opt['datafile'] = os.path.join(
            opt['datapath'], 'prosocial_dialog', self.datatype + '.json'
        )
        self.id = 'prosocial_dialog'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        f = open(path)
        self.json_data = json.load(f)
        f.close()

        for exs in self.json_data:
            texts = []
            for ex in exs:
                texts.append(ex['text'])
                if self.opt['one_turn']:
                    x = ex['text']
                else:
                    x = '\n'.join(texts)
                texts.append(ex['labels'][0])
                y = ex['safety_label']
                m = {'text': x, 'labels': y}
                yield m, True

    def num_episodes(self):
        return sum([len(x) for x in self.json_data])

    def num_examples(self):
        return sum([len(x) for x in self.json_data])


class ProsocialDialogBinarySafetyTeacher(ProsocialDialogSafetyTeacher):
    """
    Binary Safety Teacher for ProsocialDialog Data https://github.com/skywalker023/prosocial-dialog
    Casual is __ok__ and Needs Caution and Needs Intervention is __notok__
    """

    def setup_data(self, path):
        print('loading: ' + path)
        f = open(path)
        self.json_data = json.load(f)
        f.close()

        for exs in self.json_data:
            texts = []
            for ex in exs:
                texts.append(ex['text'])
                if self.opt['one_turn']:
                    x = ex['text']
                else:
                    x = '\n'.join(texts)
                texts.append(ex['labels'][0])
                y = "__ok__" if ex['safety_label'] == "__casual__" else "__notok__"
                m = {'text': x, 'labels': y}
                yield m, True


class ProsocialDialogTeacher(DialogTeacher):
    """
    Teacher for ProsocialDialog Data https://github.com/skywalker023/prosocial-dialog
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        self.datatype = opt['datatype'].split(':')[0]
        opt['datafile'] = os.path.join(
            opt['datapath'], 'prosocial_dialog', self.datatype + '.json'
        )
        self.id = 'prosocial_dialog'
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        f = open(path)
        self.json_data = json.load(f)
        f.close()

        for exs in self.json_data:
            for ex in exs:
                x = ex['text']
                y = ex['labels']
                m = {'text': x, 'labels': y}
                yield m, ex['episode_done']

    def num_episodes(self):
        return len(self.json_data)

    def num_examples(self):
        return sum([len(x) for x in self.json_data])


class DefaultTeacher(ProsocialDialogTeacher):
    pass
