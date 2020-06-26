#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from parlai.core.teachers import FixedDialogTeacher
from .build import build

import json
import os


def _path(opt):
    # build the data if it does not exist
    build(opt)

    return os.path.join(opt['datapath'], 'holl_e')


def list_to_str(lst):
    s = ''
    for ele in lst:
        if s:
            s += ' ' + ele
        else:
            s = ele
    return s


class HollETeacher(FixedDialogTeacher):
    """
    Sequence of utterances and responses with background knowledge about movies.

    From the Holl-E dataset. More information found at
    https://github.com/nikitacs16/Holl-E.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        group = argparser.add_argument_group('Holl-E Knowledge arguments')
        group.add_argument(
            '--knowledge-types',
            '-kt',
            type=str,
            default='full',
            help='Either "full" (all of the following), "none" (knowledge not used) or comma separated list of knowledge types to include. Possible types: plot, review, comments, fact_table (contains awards, taglines, and similar movies). e.g. -kt plot,review,fact_table or -kt full',
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        holle_path = _path(opt)
        self.datatype = opt['datatype'].split(':')[0]
        self.id = 'holl_e'
        if shared is not None:
            self.episodes = shared['episodes']
        else:
            self.episodes = self.setup_data(holle_path)
        self.reset()

    def setup_data(self, path):
        # use test json if valid is given
        json_dtype = self.datatype if not self.datatype.startswith('valid') else 'test'
        json_path = os.path.join(path, f'raw_{json_dtype}_data.json')
        with open(json_path) as f:
            data = json.load(f)
        episodes = []
        prev_id = None
        episode = None, []
        for d in data:
            utterance = {'query': d['query'], 'response': d['response']}
            if d['chat_id'] == prev_id:
                episode[1].append(utterance)
            else:
                if episode[1]:
                    episodes.append(episode)
                knowledge = self.get_knowledge(d)
                if knowledge.endswith('EOD'):
                    knowledge = knowledge[:-3]
                episode = knowledge, [utterance]
                prev_id = d['chat_id']
        if self.datatype.startswith('valid'):
            episodes = episodes[: len(episodes) // 2]
        elif self.datatype.startswith('test'):
            episodes = episodes[len(episodes) // 2 :]
        return episodes

    def share(self):
        shared = super().share()
        shared['episodes'] = self.episodes
        return shared

    def get_knowledge(self, data):
        ktypes = self.opt['knowledge_types'].split(',')
        if 'full' in ktypes or len(ktypes) >= 4:
            return data['full']
        elif 'none' in ktypes:
            return ''
        else:
            data = data['all_documents']
            ktype_order = {'plot': 0, 'review': 1, 'comments': 2, 'fact_table': 3}
            ktypes.sort(key=lambda x: ktype_order[x])
            knowledge = ''
            for ktype in ktypes:
                if ktype == 'fact_table':
                    fact_table = data['fact_table']
                    ft_str = ''
                    if 'box_office' in fact_table:
                        ft_str += ' ' + str(fact_table['box_office'])
                    if 'taglines' in fact_table:
                        ft_str += ' ' + list_to_str(fact_table['taglines'])
                    if 'awards' in fact_table:
                        ft_str += ' ' + list_to_str(fact_table['awards'])
                    if 'similar_movies' in fact_table:
                        ft_str += ' ' + list_to_str(fact_table['similar_movies'])
                    knowledge += '\n' + ft_str[1:]
                elif ktype == 'comments':
                    knowledge += '\n' + list_to_str(data['comments'])
                else:
                    knowledge += '\n' + data[ktype]
        return knowledge[1:]

    def num_examples(self):
        examples = 0
        for _, episode in self.episodes:
            examples += len(episode) // 2
        return examples

    def num_episodes(self):
        return len(self.episodes)

    def get(self, episode_idx, entry_idx=0):
        knowledge, episode = self.episodes[episode_idx]
        # get every other entry so we don't overlap text with a response
        text_idx = entry_idx * 2
        entry = episode[text_idx]
        episode_done = text_idx >= len(episode) - 2
        if self.opt['knowledge_types'] != 'none' and entry_idx == 0:
            text = knowledge + '\n' + entry['query']
        else:
            text = entry['query']
        action = {
            'id': self.id,
            'text': text,
            'episode_done': episode_done,
            'labels': [entry['response']],
        }
        return action


class DefaultTeacher(HollETeacher):
    pass
