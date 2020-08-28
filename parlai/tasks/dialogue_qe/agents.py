#!/usr/bin/env python3

# Copyright (c) 2017-present, Moscow Institute of Physics and Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        self.data_path = DefaultTeacher._path(opt)
        opt['datafile'] = self.data_path
        self.id = 'DialogueQE'
        self.dialogs = None
        super().__init__(opt, shared)

    @staticmethod
    def _path(opt):
        import os
        import sys
        from parlai.tasks.dialogue_qe.build import build

        build(opt)
        dt = opt['datatype'].split(':')[0]

        if dt == 'train':
            path = os.path.join(opt['datapath'], 'DialogueQE', 'train.json')
        elif dt == 'test':
            path = os.path.join(opt['datapath'], 'DialogueQE', 'test.json')
        elif dt == 'valid':
            print('warning: validation is not supporting', file=sys.stderr)
            path = None
        else:
            raise RuntimeError('Not valid datatype.')

        return path

    @staticmethod
    def _transform_utterance(utterance, user_types):
        uid = utterance['userId']
        t = user_types[uid]
        return ': '.join([utterance['userId'] + '(' + t + ')', utterance['text']])

    def setup_data(self, path):
        import json
        from functools import reduce

        print('loading: ' + path)

        if path is None:
            return iter(())

        with PathManager.open(path) as data_file:
            self.dialogs = json.load(data_file)

        for dialog in self.dialogs:
            if len(dialog['thread']) == 0:
                continue
            user_types = dict(map(lambda u: (u['id'], u['userType']), dialog['users']))
            str_threads = [
                i
                for i in map(
                    lambda u: DefaultTeacher._transform_utterance(u, user_types),
                    dialog["thread"],
                )
            ]
            dialog_txt = reduce(
                lambda u1, u2: u1 + '\n' + u2, str_threads, str_threads.pop(0)
            )
            e1 = dialog['evaluation'][0]
            e2 = dialog['evaluation'][1]
            label = '{0}:{1};{2}:{3}'.format(
                e1['userId'], e1['quality'], e2['userId'], e2['quality']
            )

            yield (dialog_txt, [label]), True
