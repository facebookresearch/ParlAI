#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from copy import deepcopy
from typing import Any, Dict, Union
import json
import logging
import os

from parlai.core.message import Message
from parlai.core.params import ParlaiParser, Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.data import DatatypeHelper
import parlai.tasks.jericho_world.constants as consts

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


def knowledge_graph_as_str(graph):
    if not graph:
        return consts.EMPTY_GRAPH_TOKEN

    graph_comps = []
    for s, r, o in graph:
        graph_comps.append(f'< {s} , {r} , {o}>')
    return ' ; '.join(graph_comps)


def extract_state_data(example_state: Dict) -> Dict:
    return {
        'observation': example_state['obs'],
        'loc_desc': example_state['loc_desc'],
        'surrounding_objs': example_state['surrounding_objs'],
        'valid_acts': list(example_state['valid_acts'].values()),
        'graph': example_state['graph'],
    }


class BaseJerichoWorldTeacher(DialogTeacher):
    """
    The base class that loads the games and episodes of the JerichoWorld.

    Note: do not use this class directly.
    """

    def __init__(self, opt: Opt, shared=None):
        opt = deepcopy(opt)
        opt['datafile'] = _path(opt)
        self.datatype = get_dtype(opt)
        self.id = 'JerichoWorldtBase'
        self._incld_loc = opt['include_location']
        self._incld_loc_desc = opt['include_location_description']
        self.keep_next_state = True
        super().__init__(opt, shared=shared)

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt=None) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        arg_group = parser.add_argument_group('Jericho World Args')
        arg_group.add_argument(
            '--include-location',
            type='bool',
            default=True,
            help='Whether to include the name of the location',
        )
        arg_group.add_argument(
            '--include-location-description',
            type='bool',
            default=True,
            help='Whether to include the text description of the location',
        )

    def _clean_example(self, game_step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keeps data from example that we need, and discars the rest.
        """
        example = {
            'action': game_step['action'],
            'state': extract_state_data(game_step['state']),
        }
        if self.keep_next_state:
            example['next_state'] = extract_state_data(game_step['next_state'])
        return example

    @abc.abstractmethod
    def generate_example_text(self, example: Union[Dict, Message]) -> str:
        """
        Implement this method for the expected text of your example.
        """

    @abc.abstractmethod
    def generate_example_label(self, example: Union[Dict, Message]) -> str:
        """
        Implement this method for the expected label of your example.
        """

    def setup_data(self, datafile: str):
        print(datafile)
        with open(datafile) as df:
            games_data = json.load(df)
            for game in games_data:
                for step_i, game_step in enumerate(game):
                    example = self._clean_example(game_step)
                    example['text'] = self.generate_example_text(example)
                    example['labels'] = [self.generate_example_label(example)]

                    new_episode = step_i == 0
                    yield example, new_episode


class StateToKGTeacher(BaseJerichoWorldTeacher):
    """
    Game state to the knowledge graph.
    """

    def generate_example_text(self, example: Union[Dict, Message]) -> str:
        return example['state']['loc_desc'].replace('\n', ' ').strip()

    def generate_example_label(self, example: Union[Dict, Message]) -> str:
        return knowledge_graph_as_str(example['state']['graph'])


class StateToActionTeacher(BaseJerichoWorldTeacher):
    """
    Game state to action.
    """

    def generate_example_text(self, example: Union[Dict, Message]) -> str:
        return example['state']['loc_desc'].replace('\n', ' ').strip()

    def generate_example_label(self, example: Union[Dict, Message]) -> str:
        return example['action']


class DefaultTeacher(StateToKGTeacher):
    pass
