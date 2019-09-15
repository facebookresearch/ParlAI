# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
import json


class CCPEAllTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        super().__init__(opt, shared)

        dt = opt['datatype'].split(':')[0]
        if dt != 'train':
            raise RuntimeError('Not valid datatype (only train).')

        if shared:
            self.data = shared['data']
        else:
            build(opt)
            self._setup_data()

        self.reset()

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return sum([len(x) for x in self.data])

    def _setup_data(self):

        fpath = os.path.join(self.opt['datapath'], 'CCPE', 'ccpe.json')

        with open(fpath, 'r') as infile:
            data = infile.read()
            new_data = data.replace('}\n{', '},{')
            json_data = json.loads(f'[{new_data}]')

        self.data = []

        for ep in range(len(json_data)):
            currEp = []
            for i, utterance in enumerate(json_data[ep]['utterances']):
                entry = []
                cnt = 0
                if i > 0:
                    entry.append(cnt)
                    cnt += 1
                    entry.append(json_data[ep]['utterances'][i - 1]['text'])
                    entry.append(utterance['text'])
                    entry.append(
                        json_data[ep]['utterances'][i - 1]['segments']
                        if 'segments' in json_data[ep]['utterances'][i - 1]
                        else []
                    )
                    entry.append(
                        json_data[ep]['utterances'][i]['segments']
                        if 'segments' in json_data[ep]['utterances'][i]
                        else []
                    )
                    entry.append(False)
                    currEp.append(entry)

            currEp[-1][5] = True
            self.data.append(currEp)

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        entry = ep[entry_idx]
        action = {
            'id': entry[0],
            'text': entry[1],
            'labels': [entry[2]],
            'textAnnotation': entry[3],
            'labelAnnotation': entry[4],
            'episode_done': entry[5],
        }
        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class CCPEAssistantTeacher(CCPEAllTeacher):
    def _setup_data(self):

        fpath = os.path.join(self.opt['datapath'], 'CCPE', 'ccpe.json')

        with open(fpath, 'r') as infile:
            data = infile.read()
            new_data = data.replace('}\n{', '},{')
            json_data = json.loads(f'[{new_data}]')

        self.data = []

        for ep in range(len(json_data)):
            currEp = []
            for i, utterance in enumerate(json_data[ep]['utterances']):
                entry = []
                cnt = 0
                if (
                    i > 0
                    and json_data[ep]['utterances'][i - 1]['speaker'] == "USER"
                    and utterance['speaker'] == "ASSISTANT"
                ):
                    entry.append(cnt)
                    cnt += 1
                    entry.append(json_data[ep]['utterances'][i - 1]['text'])
                    entry.append(utterance['text'])
                    entry.append(
                        json_data[ep]['utterances'][i - 1]['segments']
                        if 'segments' in json_data[ep]['utterances'][i - 1]
                        else []
                    )
                    entry.append(
                        json_data[ep]['utterances'][i]['segments']
                        if 'segments' in json_data[ep]['utterances'][i]
                        else []
                    )
                    entry.append(False)
                    currEp.append(entry)

            currEp[-1][5] = True
            self.data.append(currEp)


class DefaultTeacher(CCPEAllTeacher):
    pass
