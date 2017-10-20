# Copyright (c) 2017-present, Moscow Institute of Physics and Technology.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.dialog_teacher import DialogTeacher

class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        self.data_path = DefaultTeacher._path(opt)
        opt['datafile'] = self.data_path
        self.id = 'ConvAIChitChat'
        super().__init__(opt, shared)

    @classmethod
    def _path(cls, opt):
        import os
        from parlai.tasks.convai_chitchat.build import build
        build(opt)
        dt = opt['datatype'].split(':')[0]

        if dt == 'train':
            path = os.path.join(opt['datapath'], 'ConvAIChitChat', 'train.json')
        elif dt == 'valid':
            path = os.path.join(opt['datapath'], 'ConvAIChitChat', 'valid.json')
        elif dt == 'test':
            path = os.path.join(opt['datapath'], 'ConvAIChitChat', 'test.json')
        else:
            raise RuntimeError('Not valid datatype.')

        return path

    def setup_data(self, path):
        import json
        print('loading: ' + path)

        with open(path) as data_file:
            self.dialogs = json.load(data_file)

        for dialog in self.dialogs:
            prev_utterance = None
            for i, utterance in enumerate(dialog["thread"]):
                episode_done = False
                if i == len(dialog["thread"]) - 1:
                    episode_done = True
                clean_utterance = ': '.join([utterance['userId'], utterance['text']])
                res = (prev_utterance, [ clean_utterance ])
                prev_utterance = clean_utterance

                yield res, episode_done
