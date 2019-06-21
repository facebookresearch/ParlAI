#!/usr/bin/env python3

# Copyright (c) 2017-present, Moscow Institute of Physics and Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .build import build
from parlai.core.teachers import DialogTeacher

import json
import os


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        self.data_path = DefaultTeacher._path(opt)
        opt['datafile'] = self.data_path
        self.id = 'ConvAIChitChat'
        super().__init__(opt, shared)

    @staticmethod
    def _path(opt):
        build(opt)
        dt = opt['datatype'].split(':')[0]

        if dt == 'train':
            path = os.path.join(opt['datapath'], 'ConvAIChitChat', 'train.json')
        elif dt == 'test':
            path = os.path.join(opt['datapath'], 'ConvAIChitChat', 'test.json')
        elif dt == 'valid':
            raise RuntimeError('warning: validation is not supported')
        else:
            raise RuntimeError('Not valid datatype.')

        return path

    @staticmethod
    def _fold_utterances(raw_dialog):
        dialog = []
        for utterance in raw_dialog:
            if len(dialog) > 0 and dialog[-1]['userId'] == utterance['userId']:
                dialog[-1]['text'] = dialog[-1]['text'] + '\n' + utterance['text']
            else:
                dialog.append(
                    {'text': utterance['text'], 'userId': utterance['userId']}
                )
        return dialog

    @staticmethod
    def _create_learning_examples(opponent_utterances, answer_utterances):
        examples = [
            u
            for u in map(
                lambda pair: ((pair[0]['text'], [pair[1]['text']]), False),
                zip(opponent_utterances, answer_utterances),
            )
        ]
        return examples

    @staticmethod
    def _data_generator(dialogs_dict):
        for dialog in dialogs_dict:
            folded_dialog = DefaultTeacher._fold_utterances(dialog['thread'])
            context = dialog['context']

            if len(folded_dialog) < 2:
                continue

            u1_utterances = folded_dialog[::2]
            u2_utterances = folded_dialog[1::2]

            it = [((context, ['']), True)] + DefaultTeacher._create_learning_examples(
                u1_utterances, u2_utterances
            )
            for second_user_examples in it:
                yield second_user_examples

            if len(u1_utterances) > 1:
                examples = [
                    ((context, [u1_utterances[0]['text']]), True)
                ] + DefaultTeacher._create_learning_examples(
                    u2_utterances, u1_utterances[1:]
                )
            else:
                examples = [((context, [u1_utterances[0]['text']]), True)]

            for first_user_examples in examples:
                yield first_user_examples

    @staticmethod
    def setup_data(path):
        print('loading: ' + path)

        if path is None:
            return iter(())

        with open(path) as data_file:
            dialogs = json.load(data_file)

        return DefaultTeacher._data_generator(dialogs)
