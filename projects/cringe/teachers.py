#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.tasks.jsonfile.agents import JsonTeacher


class IterativeTeacher(JsonTeacher):
    delete_tokens = ['_POTENTIALLY_UNSAFE__']

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('IterativeTeacher options')
        agent.add_argument(
            '--prepend-classifier-label',
            type=bool,
            default=False,
            help='If true, prepend the classifier label to the generation label. This should only be used to inspect the data',
        )
        return parser

    def __init__(self, opt, shared=None):
        assert (
            'jsonfile_datapath' in opt
        ), 'You need to provide the --jsonfile-datapath flag for the IterativeTeacher.'
        super().__init__(opt, shared)

    def setup_data(self, path):
        for example, episode_end in super().setup_data(path):
            example['is_ltr'] = True
            labels = example.pop('labels')
            for word in self.delete_tokens:
                labels = [l.replace(word, '').strip() for l in labels]
            example['labels'] = [l.strip() for l in labels]

            if self.opt.get('prepend_classifier_label', False):
                example['labels'][0] = (
                    example['classifier_label'] + ': ' + example['labels'][0]
                )

            yield example, episode_end

    def _get_ep_from_turns(self, xturns, yturns):
        eps = []
        for xturn, yturn in zip(xturns, yturns):
            turn = {}
            turn['text'] = xturn.get('text').strip()
            turn['labels'] = [yturn.get('text').strip()]
            if 'pos_classifier_prediction' in yturn['metrics']:
                class_label_int = int(yturn['metrics']['pos_classifier_prediction'])
                turn['classifier_label'] = 'neg' if class_label_int == 0 else 'pos'
            elif 'classifier_accuracy' in yturn['metrics']:
                class_label_int = int(yturn['metrics']['classifier_accuracy'])
                turn['classifier_label'] = 'neg' if class_label_int == 0 else 'pos'
            elif 'f1' in yturn['metrics']:
                turn['classifier_label'] = (
                    'neg' if float(yturn['metrics']['f1']) == 0.0 else 'pos'
                )
            eps.append(turn)
        return eps
