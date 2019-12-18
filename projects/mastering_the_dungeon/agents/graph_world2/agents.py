#!/usr/bin/env python3

##
## Copyright (c) Facebook, Inc. and its affiliates.
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##

from parlai.core.agents import Agent
from collections import defaultdict as dd
import spacy
from .models import ObjectChecklistModel, Seq2SeqModel
import numpy as np
from torch.autograd import Variable
import torch
from copy import deepcopy
from projects.mastering_the_dungeon.tasks.graph_world2.graph import (
    DEFAULT_ROOMS,
    DEFAULT_CONTAINERS,
    DEFAULT_AGENTS,
    DEDUP_OBJECTS,
    DEDUP_PROPS,
)

nlp = spacy.load('en')


def parse_action_tuple(insts):
    if insts[0] in [
        'go',
        'drop',
        'wear',
        'wield',
        'eat',
        'drink',
        'remove',
        'unwield',
        'hit',
    ]:
        return insts[0], ' '.join(insts[1:])
    if insts[0] == 'get':
        args = ' '.join(insts[1:]).split(' from ')
        if len(args) == 1:
            return 'get', args[0]
        else:
            return 'get', args[0], args[1]
    if insts[0] == 'give':
        args = ' '.join(insts[1:]).split(' to ')
        return 'give', args[0], args[1]
    if insts[0] == 'take':
        args = ' '.join(insts[1:]).split(' from ')
        return 'take', args[0], args[1]
    if insts[0] == 'put':
        args = ' '.join(insts[1:]).split(' in ')
        return 'put', args[0], args[1]
    assert False, insts


def reverse_parse_action(action_tuple):
    if action_tuple[0] == 'stop':
        return 'STOP'
    if action_tuple[0] in [
        'go',
        'drop',
        'wear',
        'wield',
        'eat',
        'drink',
        'remove',
        'unwield',
        'hit',
    ]:
        return '{} {}'.format(action_tuple[0], action_tuple[1])
    if action_tuple[0] == 'get':
        if len(action_tuple) == 2:
            return 'get {}'.format(action_tuple[1])
        else:
            return 'get {} from {}'.format(action_tuple[1], action_tuple[2])
    if action_tuple[0] == 'give':
        return 'give {} to {}'.format(action_tuple[1], action_tuple[2])
    if action_tuple[0] == 'take':
        return 'take {} from {}'.format(action_tuple[1], action_tuple[2])
    if action_tuple[0] == 'put':
        return 'put {} in {}'.format(action_tuple[1], action_tuple[2])
    assert False, action_tuple


class DataAgentBase(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        if not shared:
            self.word2cnt = dd(int)
        else:
            self.word2cnt = shared['word2cnt']

    def _tokenize(self, text, lower=True):
        return list(map(lambda x: x.lower_ if lower else x.orth_, list(nlp(text))))

    def act(self):
        observation = self.observation

        tokens = self._tokenize(observation['text'])
        for token in tokens:
            self.word2cnt[token] += 1
        return {}

    def build(self):
        opt = self.opt
        word2cnt = [(k, v) for k, v in self.word2cnt.items()]
        word2cnt.sort(key=lambda x: x[1], reverse=True)
        word_offset, word2index = 2, {}
        word2index['PAD'] = 0
        word2index['UNK'] = 1
        for i in range(opt['vocab_size'] - word_offset):
            if i >= len(word2cnt):
                break
            word = word2cnt[i][0]
            word2index[word] = i + word_offset
        self.word2index = word2index
        self.wordcnt = len(word2index)

    def _get_word_index(self, token):
        if token in self.word2index:
            return self.word2index[token]
        return self.word2index['UNK']

    def share(self):
        shared = super().share()
        shared['word2cnt'] = self.word2cnt
        return shared

    def build_action_id(self):
        action2id = {}
        offset = 0
        for ent in DEFAULT_ROOMS:
            action2id[('go', ent)] = offset
            offset += 1

        for ent in DEDUP_OBJECTS + DEFAULT_CONTAINERS:
            action2id[('get', ent)] = offset
            offset += 1
            action2id[('drop', ent)] = offset
            offset += 1

        for i, ent in enumerate(DEDUP_OBJECTS):
            if DEDUP_PROPS[i] == 'food':
                action2id[('eat', ent)] = offset
                offset += 1
            elif DEDUP_PROPS[i] == 'drink':
                action2id[('drink', ent)] = offset
                offset += 1
            elif DEDUP_PROPS[i] == 'wearable':
                action2id[('wear', ent)] = offset
                offset += 1
                action2id[('remove', ent)] = offset
                offset += 1
            elif DEDUP_PROPS[i] == 'wieldable':
                action2id[('wield', ent)] = offset
                offset += 1
                action2id[('unwield', ent)] = offset
                offset += 1

        for ent_i in DEDUP_OBJECTS + DEFAULT_CONTAINERS:
            for ent_j in DEFAULT_CONTAINERS:
                if ent_i == ent_j:
                    continue
                action2id[('put', ent_i, ent_j)] = offset
                offset += 1
                action2id[('get', ent_i, ent_j)] = offset
                offset += 1

        for ent_i in DEDUP_OBJECTS + DEFAULT_CONTAINERS:
            for ent_j in DEFAULT_AGENTS:
                if ent_j == 'dragon':
                    continue
                action2id[('give', ent_i, ent_j)] = offset
                offset += 1
                action2id[('take', ent_i, ent_j)] = offset
                offset += 1

        for ent in DEFAULT_AGENTS:
            if ent != 'dragon':
                action2id[('hit', ent)] = offset
                offset += 1

        action2id[('stop',)] = offset
        offset += 1

        self.y_dim = offset
        print('y_dim = {}'.format(self.y_dim))

        self.action2id = action2id
        self.id2action = [None for _ in range(self.y_dim)]
        for k, v in self.action2id.items():
            self.id2action[v] = k

    def build_action_key(self):
        action_key = np.zeros((self.y_dim,), dtype=np.int64)
        for i in range(self.y_dim):
            action_tuple = self.get_action_tuple(i)
            if len(action_tuple) <= 1:
                continue
            my_key = action_tuple[1]
            action_key[i] = self._get_word_index(my_key.replace(' ', '_'))
        self.action_key = action_key

    def build_second_action_key(self):
        second_action_key = np.zeros((self.y_dim,), dtype=np.int64)
        for i in range(self.y_dim):
            action_tuple = self.get_action_tuple(i)
            if len(action_tuple) <= 2:
                continue
            my_key = action_tuple[2]
            second_action_key[i] = self._get_word_index(my_key.replace(' ', '_'))
        self.second_action_key = second_action_key

    def build_action_type(self):
        action_types = deepcopy(self.ACTION_TYPES)
        action_type = np.zeros((self.y_dim,), dtype=np.int64)
        for i in range(self.y_dim):
            action_tuple = self.get_action_tuple(i)
            my_type = action_tuple[0]
            action_type[i] = action_types.index(my_type)
        self.action_type = action_type
        self.num_actions = len(action_types)

    def get_num_actions(self):
        return self.num_actions

    def build_check_mapping(self):
        check_to_key = {}
        for i in range(self.y_dim):
            action_tuple = self.get_action_tuple(i)
            if len(action_tuple) == 1:
                check_to_key[action_tuple] = action_tuple[0]
            else:
                check_to_key[action_tuple] = action_tuple[1]
        key_to_check = dd(set)
        for k, v in check_to_key.items():
            key_to_check[v].add(k)
        self.check_to_key, self.key_to_check = check_to_key, key_to_check

        check_mapping = np.zeros((self.y_dim, self.y_dim), dtype=np.float32)
        for i in range(self.y_dim):
            for j in range(self.y_dim):
                if (
                    self.get_action_tuple(j)
                    in key_to_check[check_to_key[self.get_action_tuple(i)]]
                ):
                    check_mapping[i, j] = 1.0
        self.check_mapping = check_mapping

    def get_check_mapping(self):
        return self.check_mapping

    def get_action_tuple(self, id):
        return self.id2action[id]

    def get_action_id(self, action):
        return self.action2id[action]

    def reverse_parse_action(self, action_tuple):
        return reverse_parse_action(action_tuple)

    def get_mask(self, g, mask):
        possible_actions = g.get_possible_actions()
        for action in possible_actions:
            action_tuple = parse_action_tuple(action.split())
            action_id = self.get_action_id(action_tuple)
            mask[action_id] = 1.0
        mask[self.get_action_id(('stop',))] = 1.0


class ObjectChecklistDataAgent(DataAgentBase):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.num_rooms = len(DEFAULT_ROOMS)
        self.num_objects = len(DEDUP_OBJECTS)
        self.num_containers = len(DEFAULT_CONTAINERS)
        self.num_npcs = len(DEFAULT_AGENTS) - 1

    def build(self):
        self.ACTION_TYPES = [
            'go',
            'get',
            'drop',
            'eat',
            'drink',
            'wear',
            'wield',
            'remove',
            'unwield',
            'give',
            'take',
            'put',
            'hit',
            'stop',
        ]
        super().build()
        self.build_action_id()
        self.build_action_key()
        self.build_second_action_key()
        self.build_action_type()
        self.build_check_mapping()

    def get_room(self, g):
        return self._get_word_index(
            g.node_to_desc_raw(g.node_contained_in('dragon')).replace(' ', '_')
        )

    def _tokenize(self, text, lower=True):
        tokenized = ' '.join(
            list(map(lambda x: x.lower_ if lower else x.orth_, list(nlp(text))))
        )
        for ent in DEFAULT_ROOMS + DEFAULT_CONTAINERS + DEFAULT_AGENTS + DEDUP_OBJECTS:
            tokenized = tokenized.replace(ent, ent.replace(' ', '_'))
        return tokenized.split()

    def get_data(self, observations, datatype='train', assert_=True):
        opt = self.opt
        batch_size = len(observations)
        seq_in, seq_out = 0, 0
        tokens_list, inst_list, symb_points_list = [], [], []
        for observation in observations:
            graph, text, actions = (
                observation['graph'],
                observation['text'],
                observation['actions'],
            )
            tokens_list.append(self._tokenize(text))
            seq_in = max(seq_in, len(tokens_list[-1]))

            graph = observation['graph']
            inst, symb_points = graph.parse(actions)
            seq_out = max(seq_out, len(symb_points) - 1 + 1)  # +1 for stop
            inst_list.append(inst)
            symb_points_list.append(symb_points)

        if datatype == 'valid':
            seq_out = opt['max_seq_out']

        seq_in = min(seq_in, opt['max_seq_in'])
        y_dim = self.y_dim
        x = np.zeros((batch_size, seq_in), dtype=np.int64)
        current_room = np.zeros((batch_size, seq_out), dtype=np.int64)
        checked = np.zeros((batch_size, seq_out + 1, y_dim), dtype=np.float32)
        y = np.zeros((batch_size, seq_out, y_dim), dtype=np.float32)
        y_mask = np.zeros((batch_size, seq_out, y_dim), dtype=np.float32)
        counter_feat = np.zeros((batch_size, seq_out, y_dim), dtype=np.int64)

        graph = observations[0]['graph']

        action_key = self.action_key
        action_type = self.action_type
        second_action_key = self.second_action_key

        for i in range(batch_size):
            for j, token in enumerate(tokens_list[i]):
                if j >= seq_in:
                    break
                x[i, j] = self._get_word_index(token)

            inst = inst_list[i]
            g = observations[i]['graph'].copy()
            len_plus_one = len(symb_points_list[i])

            action_tuples = []
            for j in range(len_plus_one - 1):
                k, l = symb_points_list[i][j], symb_points_list[i][j + 1]
                action_tuples.append(parse_action_tuple(inst[k:l]))

            for j in range(len_plus_one):
                if j < len_plus_one - 1:
                    cur_tuple = action_tuples[j]
                    y[i, j, self.get_action_id(cur_tuple)] = 1.0
                else:
                    stop_tuple = ('stop',)
                    y[i, j, self.get_action_id(stop_tuple)] = 1.0

                current_room[i, j] = self.get_room(g)

                self.get_mask(g, y_mask[i, j])

                if j < len_plus_one - 1:
                    k, l = symb_points_list[i][j], symb_points_list[i][j + 1]
                    parse_success = g.parse_exec(' '.join(inst[k:l]))
                    if assert_:
                        assert parse_success, (
                            ' '.join(inst[k:l]) + '  ' + ' '.join(inst)
                        )

                    counter_feat[i, j + 1] = counter_feat[i, j]
                    cur_tuple = action_tuples[j]
                    for action_name in self.key_to_check[self.check_to_key[cur_tuple]]:
                        action_id = self.get_action_id(action_name)
                        counter_feat[i, j + 1, action_id] += 1

        counter_feat = np.clip(counter_feat, None, opt['counter_max'])
        return (
            x,
            action_key,
            second_action_key,
            action_type,
            current_room,
            checked,
            y,
            y_mask,
            counter_feat,
        )


class Seq2SeqDataAgent(DataAgentBase):
    def build(self):
        self.ACTION_TYPES = [
            'go',
            'get',
            'drop',
            'eat',
            'drink',
            'wear',
            'wield',
            'remove',
            'unwield',
            'give',
            'take',
            'put',
            'hit',
            'stop',
        ]
        super().build()
        self.build_action_id()

    def get_data(self, observations, datatype='train', assert_=True):
        opt = self.opt
        batch_size = len(observations)
        seq_in, seq_out = 0, 0
        tokens_list, inst_list, symb_points_list = [], [], []
        for observation in observations:
            graph, text, actions = (
                observation['graph'],
                observation['text'],
                observation['actions'],
            )
            tokens_list.append(self._tokenize(text))
            seq_in = max(seq_in, len(tokens_list[-1]))

            graph = observation['graph']
            inst, symb_points = graph.parse(actions)
            seq_out = max(seq_out, len(symb_points) - 1 + 1)  # +1 for stop
            inst_list.append(inst)
            symb_points_list.append(symb_points)

        if datatype == 'valid':
            seq_out = opt['max_seq_out']

        seq_in = min(seq_in, opt['max_seq_in'])
        y_dim = self.y_dim
        x = np.zeros((batch_size, seq_in), dtype=np.int64)
        y = np.zeros((batch_size, seq_out, y_dim), dtype=np.float32)

        for i in range(batch_size):
            for j, token in enumerate(tokens_list[i]):
                if j >= seq_in:
                    break
                x[i, j] = self._get_word_index(token)

            inst = inst_list[i]
            g = observations[i]['graph'].copy()
            len_plus_one = len(symb_points_list[i])

            action_tuples = []
            for j in range(len_plus_one - 1):
                k, l = symb_points_list[i][j], symb_points_list[i][j + 1]
                action_tuples.append(parse_action_tuple(inst[k:l]))

            for j in range(len_plus_one):
                if j < len_plus_one - 1:
                    cur_tuple = action_tuples[j]

                    y[i, j, self.get_action_id(cur_tuple)] = 1.0

                else:
                    stop_tuple = ('stop',)
                    y[i, j, self.get_action_id(stop_tuple)] = 1.0

                if j < len_plus_one - 1:
                    k, l = symb_points_list[i][j], symb_points_list[i][j + 1]
                    parse_success = g.parse_exec(' '.join(inst[k:l]))
                    if assert_:
                        assert parse_success, ' '.join(inst[k:l])
        return x, y


class ModelAgentBase(Agent):
    def __init__(self, opt, shared=None, data_agent=None):
        super().__init__(opt, shared)
        if not shared:
            self.data_agent = data_agent
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = torch.optim.Adam(params, lr=opt['lr'])
            if opt['cuda']:
                self.model.cuda()
        else:
            self.data_agent = shared['data_agent']
            self.model = shared['model']
            self.optimizer = shared['optimizer']

    def share(self):
        shared = super().share()
        shared['data_agent'] = self.data_agent
        shared['model'] = self.model
        shared['optimizer'] = self.optimizer
        return shared

    def _get_variable(self, np_a, volatile=False):
        if self.opt['cuda']:
            return Variable(torch.from_numpy(np_a).cuda(), volatile=volatile)
        return Variable(torch.from_numpy(np_a), volatile=volatile)

    def _get_f1(self, tokens_1, tokens_2):
        tokens_1, tokens_2 = set(tokens_1), set(tokens_2)
        tp, fp, fn = 0, 0, 0
        for token in tokens_2:
            if token in tokens_1:
                tp += 1
            else:
                fp += 1
        for token in tokens_1:
            if token not in tokens_2:
                fn += 1
        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        return f1

    def act(self):
        return self.batch_act([self.observation])[0]


class ObjectChecklistModelAgent(ModelAgentBase):
    def __init__(self, opt, shared=None, data_agent=None):
        if not shared:
            self.model = ObjectChecklistModel(opt, data_agent)
        super().__init__(opt, shared, data_agent)

    def batch_act(self, observations):
        ori_len = len(observations)
        observations = [obv for obv in observations if 'text' in obv]
        if self.opt['datatype'] == 'train' or self.opt['datatype'] == 'pretrain':
            (
                x,
                action_key,
                second_action_key,
                action_type,
                current_room,
                checked,
                y,
                y_mask,
                counter_feat,
            ) = self.data_agent.get_data(observations)
            (
                x,
                action_key,
                second_action_key,
                action_type,
                current_room,
                checked,
                y,
                y_mask,
                counter_feat,
            ) = (
                self._get_variable(x),
                self._get_variable(action_key),
                self._get_variable(second_action_key),
                self._get_variable(action_type),
                self._get_variable(current_room),
                self._get_variable(checked),
                self._get_variable(y),
                self._get_variable(y_mask),
                self._get_variable(counter_feat),
            )

            loss = self.model.forward_loss(
                x,
                action_key,
                second_action_key,
                action_type,
                current_room,
                checked,
                y,
                y_mask,
                counter_feat,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            reply = [{'loss': loss.data[0]} for _ in range(ori_len)]
            return reply
        else:
            (
                x,
                action_key,
                second_action_key,
                action_type,
                current_room,
                checked,
                y,
                y_mask,
                counter_feat,
            ) = self.data_agent.get_data(observations, 'valid')
            (
                x,
                action_key,
                second_action_key,
                action_type,
                current_room,
                checked,
                y,
                y_mask,
                counter_feat,
            ) = (
                self._get_variable(x, True),
                self._get_variable(action_key, True),
                self._get_variable(second_action_key, True),
                self._get_variable(action_type, True),
                self._get_variable(current_room, True),
                self._get_variable(checked, True),
                self._get_variable(y, True),
                self._get_variable(y_mask, True),
                self._get_variable(counter_feat),
            )

            loss = self.model.forward_loss(
                x,
                action_key,
                second_action_key,
                action_type,
                current_room,
                checked,
                y,
                y_mask,
                counter_feat,
                False,
            )
            reply = [
                {
                    'loss': 0.0,
                    'cnt': 0.0,
                    'acc': 0,
                    'len': 0,
                    'f1': 0,
                    'correct_data': [],
                    'wrong_data': [],
                }
                for _ in range(ori_len)
            ]

            check_mapping = self.data_agent.get_check_mapping()
            check_mapping = self._get_variable(check_mapping, True)
            text_out = self.model.forward_predict(
                x,
                action_key,
                second_action_key,
                action_type,
                check_mapping,
                checked,
                [obv['graph'] for obv in observations],
                self.data_agent,
            )

            for i in range(len(observations)):
                data_rep = '{} ||| {} ||| {}'.format(
                    observations[i]['actions'],
                    ' '.join(text_out[i][:-1]),
                    observations[i]['text'],
                )

                graph_a, graph_b = (
                    observations[i]['graph'].copy(),
                    observations[i]['graph'].copy(),
                )
                graph_a.parse_exec(observations[i]['actions'])
                graph_b.parse_exec(' '.join(text_out[i][:-1]))
                if graph_a == graph_b:
                    reply[i]['acc'] = 1.0
                    reply[i]['correct_data'].append(data_rep)
                else:
                    reply[i]['wrong_data'].append(data_rep)

                inst, symb_points = observations[i]['graph'].parse(
                    observations[i]['actions']
                )
                text_gt = []
                for j in range(len(symb_points) - 1):
                    k, l = symb_points[j], symb_points[j + 1]
                    text_gt.append(' '.join(inst[k:l]))
                reply[i]['f1'] = self._get_f1(text_gt, text_out[i])

                reply[i]['loss'] = loss.data[0]
                reply[i]['cnt'] = observations[i]['weight']
                reply[i]['len'] = len(text_gt)

            return reply


class Seq2SeqModelAgent(ModelAgentBase):
    def __init__(self, opt, shared=None, data_agent=None):
        if not shared:
            self.model = Seq2SeqModel(opt, data_agent)
        super().__init__(opt, shared, data_agent)

    def batch_act(self, observations):
        ori_len = len(observations)
        observations = [obv for obv in observations if 'text' in obv]
        if self.opt['datatype'] == 'train':
            x, y = self.data_agent.get_data(observations)
            x, y = self._get_variable(x), self._get_variable(y)

            loss = self.model.forward_loss(x, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            reply = [{}] * ori_len
            reply[0]['loss'] = loss.data[0]
            return reply
        else:
            x, y = self.data_agent.get_data(observations, 'valid')
            x, y = self._get_variable(x), self._get_variable(y)

            loss = self.model.forward_loss(x, y)
            reply = [
                {
                    'loss': 0.0,
                    'cnt': 0.0,
                    'acc': 0,
                    'len': 0,
                    'f1': 0,
                    'correct_data': [],
                    'wrong_data': [],
                }
                for _ in range(ori_len)
            ]

            text_out = self.model.forward_predict(
                x, [obv['graph'] for obv in observations], self.data_agent
            )

            for i in range(len(observations)):
                data_rep = '{} ||| {} ||| {}'.format(
                    observations[i]['actions'],
                    ' '.join(text_out[i][:-1]),
                    observations[i]['text'],
                )

                graph_a, graph_b = (
                    observations[i]['graph'].copy(),
                    observations[i]['graph'].copy(),
                )
                graph_a.parse_exec(observations[i]['actions'])
                graph_b.parse_exec(' '.join(text_out[i][:-1]))
                if graph_a == graph_b:
                    reply[i]['acc'] = 1.0
                    reply[i]['correct_data'].append(data_rep)
                else:
                    reply[i]['wrong_data'].append(data_rep)

                inst, symb_points = observations[i]['graph'].parse(
                    observations[i]['actions']
                )
                text_gt = []
                for j in range(len(symb_points) - 1):
                    k, l = symb_points[j], symb_points[j + 1]
                    text_gt.append(' '.join(inst[k:l]))
                reply[i]['f1'] = self._get_f1(text_gt, text_out[i])

                reply[i]['loss'] = loss.data[0]
                reply[i]['cnt'] = observations[i]['weight']
                reply[i]['len'] = len(text_gt)

            return reply
