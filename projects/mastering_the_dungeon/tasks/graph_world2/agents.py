#!/usr/bin/env python3

##
## Copyright (c) Facebook, Inc. and its affiliates.
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
import random
from parlai.core.teachers import Teacher
from os.path import join
from collections import defaultdict as dd
from copy import deepcopy
import os
import pickle


class DefaultTeacher(Teacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        self.terminate = opt['terminate']
        self.random = not self.terminate
        self.step_size = opt.get('batchsize', 1)
        self.episode_index = shared and shared.get('batchindex') or 0
        self.opt = deepcopy(opt)

        if not shared:
            datapath = join(opt['datapath'], 'graph_world2', opt['datatype'])
            self.data = self._setup_data(datapath)
            if hasattr(self, 'valid_weights'):
                assert len(self.valid_weights) == len(self.data), (
                    len(self.valid_weights),
                    len(self.data),
                )
            self.stats = {
                'loss': 0,
                'cnt': 0,
                'acc': 0,
                'f1': 0,
                'acc_len': dd(float),
                'cnt_len': dd(float),
                'correct_data': [],
                'wrong_data': [],
            }
        else:
            self.data = shared['data']
            self.stats = shared['stats']
            if 'valid_weights' in shared:
                self.valid_weights = shared['valid_weights']
        self.len = len(self.data)
        super().__init__(opt, shared)

        self.iter = shared and shared.get('batchindex') or 0

    def __len__(self):
        return self.len

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['stats'] = self.stats
        if hasattr(self, 'valid_weights'):
            shared['valid_weights'] = self.valid_weights
        return shared

    def act(self):
        if self.episode_index >= self.len and self.terminate:
            self.epochDone = True
        if self.epochDone:
            return {'episode_done': True}
        self.iter += self.step_size
        opt = self.opt
        return_example = self.data[self.episode_index]
        if hasattr(self, 'valid_weights'):
            return_weight = self.valid_weights[self.episode_index]
        else:
            return_weight = 1.0
        self.episode_index += self.step_size
        if self.episode_index >= self.len:
            if self.terminate:
                self.epochDone = True
            self.episode_index %= self.len
            if self.random and self.episode_index == opt['batchsize'] - 1:
                random.shuffle(self.data)
        return {
            'text': return_example[2],
            'actions': return_example[3],
            'graph': return_example[1],
            'episode_done': True,
            'weight': return_weight,
        }

    def observe(self, observation):
        self.observation = observation
        if self.datatype == 'valid':
            self.stats['loss'] += observation['loss'] * observation['cnt']
            self.stats['acc'] += observation['acc'] * observation['cnt']
            self.stats['f1'] += observation['f1'] * observation['cnt']
            self.stats['cnt'] += observation['cnt']
            l = observation['len']
            self.stats['acc_len'][l] += observation['acc'] * observation['cnt']
            self.stats['cnt_len'][l] += observation['cnt']

            self.stats['correct_data'].extend(observation['correct_data'])
            self.stats['wrong_data'].extend(observation['wrong_data'])
        else:
            if 'loss' in observation:
                self.stats['loss'] += observation['loss']
                self.stats['cnt'] += 1
        return observation

    def report(self):
        if self.datatype == 'train' or self.datatype == 'pretrain':
            stats = deepcopy(self.stats)
            stats['loss'] /= stats['cnt']
            self.stats['loss'] = 0.0
            self.stats['cnt'] = 0
            return stats
        else:
            return self.stats

    def _setup_data(self, datapath):
        opt = self.opt
        if opt['weight_file'] and self.datatype == 'valid':
            self.valid_weights = pickle.load(open(opt['weight_file'], 'rb'))

        if opt['train_data_file'] != '' and self.datatype == 'train':
            return pickle.load(open(opt['train_data_file'], 'rb'))
        if opt['valid_data_file'] != '' and self.datatype == 'valid':
            return pickle.load(open(opt['valid_data_file'], 'rb'))

        data = []
        for filename in os.listdir(datapath):
            if filename.endswith('pkl'):
                loaded_data = pickle.load(open(join(datapath, filename), 'rb'))
                data.append(loaded_data)
        if self.random:
            random.shuffle(data)
        return data
