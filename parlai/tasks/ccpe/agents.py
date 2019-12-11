#!/usr/bin/env python3
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

        flattenedData = []

        for ep in range(len(json_data)):
            currEp = []

            entry = {}
            currSegments = []
            for i, utterance in enumerate(json_data[ep]['utterances']):
                if (
                    i < len(json_data[ep]['utterances']) - 1
                    and json_data[ep]['utterances'][i + 1]['speaker']
                    == utterance['speaker']
                ):
                    json_data[ep]['utterances'][i + 1]['text'] = (
                        utterance['text']
                        + '\n'
                        + json_data[ep]['utterances'][i + 1]['text']
                    )
                    currSegments.append(
                        utterance['segments'] if 'segments' in utterance else []
                    )
                    continue

                if (
                    i == len(json_data[ep]['utterances']) - 1
                    or json_data[ep]['utterances'][i + 1]['speaker']
                    != utterance['speaker']
                ):
                    entry['speaker'] = utterance['speaker']
                    entry['text'] = utterance['text']
                    currSegments.append(
                        utterance['segments']
                    ) if 'segments' in utterance else currSegments.append([])
                    entry['segments'] = currSegments
                    currEp.append(entry)
                    entry = {}
                    currSegments = []

            flattenedData.append(currEp)

        self.userData = []
        self.assistantData = []

        for ep in range(len(flattenedData)):
            currUserEp = []
            currAssistantEp = []

            userCnt = 0
            asssistantCnt = 0
            for i, currUtt in enumerate(flattenedData[ep]):
                if i > 0:
                    if (
                        currUtt['speaker'] == 'USER'
                        and flattenedData[ep][i - 1]['speaker'] == 'ASSISTANT'
                    ):
                        entry = []
                        entry.append(userCnt)
                        entry.append(currUtt['text'])
                        entry.append([flattenedData[ep][i - 1]['text']])
                        entry.append(currUtt['segments'])
                        entry.append(flattenedData[ep][i - 1]['segments'])
                        entry.append(False)
                        currUserEp.append(entry)
                        userCnt += 1
                    if (
                        currUtt['speaker'] == 'ASSISTANT'
                        and flattenedData[ep][i - 1]['speaker'] == 'USER'
                    ):
                        entry = []
                        entry.append(asssistantCnt)
                        entry.append(currUtt['text'])
                        entry.append([flattenedData[ep][i - 1]['text']])
                        entry.append(currUtt['segments'])
                        entry.append(flattenedData[ep][i - 1]['segments'])
                        entry.append(False)
                        currAssistantEp.append(entry)
                        asssistantCnt += 1

            currUserEp[-1][5] = True
            currAssistantEp[-1][5] = True
            self.userData.append(currUserEp)
            self.assistantData.append(currAssistantEp)

        self.data = self.assistantData + self.userData

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        entry = ep[entry_idx]
        action = {
            'id': entry[0],
            'text': entry[1],
            'labels': entry[2],
            'textSegments': entry[3],
            'labelSegments': entry[4],
            'episode_done': entry[5],
        }
        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class CCPEAssistantTeacher(CCPEAllTeacher):
    def _setup_data(self):
        super()._setup_data()
        self.data = self.assistantData


class DefaultTeacher(CCPEAllTeacher):
    pass
