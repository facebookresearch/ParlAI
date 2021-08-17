#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import json
import logging
import os

from parlai.core.teachers import DialogTeacher
from parlai.core.opt import Opt
from parlai.utils.data import DatatypeHelper


from .build import build, DATASET_NAME


def get_dtype(opt):
    return DatatypeHelper.fold(opt.get('datatype', 'train'))


def _path(opt):
    build(opt)
    dpath = os.path.join(opt['datapath'], DATASET_NAME)
    dtype = get_dtype(opt)
    if dtype == 'valid':
        logging.warning(
            'This data set does not have valid split. Using `test` instead.'
        )
        dtype = 'test'
    return os.path.join(dpath, f'{dtype}.json')


def knowledge_graph_as_label(graph):
    graph_comps = []
    for s, r, o in graph:
        graph_comps.append(f'< {s} , {r} , {o}>')
    return ' ; '.join(graph_comps)


class BaseJerichoWorldTeacher(DialogTeacher):
    """
    The base class that loads the games and episodes of the JerichoWorld
    
    Note: do not use this class directly.
    """

    def __init__(self, opt: Opt, shared=None):
        opt = deepcopy(opt)
        opt['datafile'] = _path(opt)
        self.datatype = get_dtype(opt)
        self.id = 'JerichoWorldtBase'
        super().__init__(opt, shared=shared)

    def setup_data(self, datafile: str):
        print(datafile)
        with open(datafile) as df:
            games_data = json.load(df)
            for game in games_data:
                for step_i, step in enumerate(game):
                    new_episode = step_i == 0
                    yield step, new_episode


class StateToKG(BaseJerichoWorldTeacher):
    """
    The game state to the knowledge graph teacher.
    """

    def setup_data(self, datafile: str):
        for old_example, new_episode in super().setup_data(datafile):
            # Just keeping what we need
            game_state = old_example['state']
            text = game_state['loc_desc'].strip().replace('\n', ' ')
            label = knowledge_graph_as_label(game_state['graph'])
            example = {'text': text, 'labels': [label]}
            yield example, new_episode


class DefaultTeacher(StateToKG):
    pass
