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


def clean_text(text: str) -> str:
    """
    Removes extra spaces and new lines from the text.
    """
    return text.replace('\n', ' ').strip()


def wrap_content(content: str, content_type: str) -> str:
    """
    Wraps content in tokens that shows its beginning and the end.
    """
    s = f'__{content_type}__'
    e = f'__end-{content_type}__'
    return f'{s} {content} {e}'


def knowledge_graph_as_str(graph):
    if not graph:
        return consts.EMPTY_GRAPH_TOKEN

    graph_comps = []
    for s, r, o in graph:
        graph_comps.append(f'< {s} , {r} , {o}>')
    return consts.SET_MEMBERS_DELIM.join(graph_comps)


def extract_state_data(example_state: Dict, delim: str = ' ') -> Dict:
    def concat_vals(d):
        return [delim.join(p) for p in d.values()]

    return {
        'location_name': example_state['location']['name'],
        'observation': example_state['obs'],
        'location_desc': example_state['loc_desc'],
        'surrounding_objs': concat_vals(example_state['surrounding_objs']),
        'inventory_objs': concat_vals(example_state['inv_objs']),
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
        self._incld_loc_name = opt['include_location']
        self.delim = opt['delimiter']
        self._incld_loc_desc = opt['include_location_description']
        self._incld_surr_objs = opt['include_surrounding_objects']
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
        arg_group.add_argument(
            '--include-surrounding-objects',
            type='bool',
            default=True,
            help='Whether to include the list of surrounding objects',
        )
        arg_group.add_argument(
            '--delimiter',
            type=str,
            default='\n',
            help='Delimiter string to use between features',
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

    def location_context(self, example_state: Dict) -> str:
        """
        Generates the context text for the location.
        """
        loc_context = []

        if self._incld_loc_name:
            loc_context.append(
                wrap_content(example_state['location_name'], consts.LOCATION_NAME)
            )

        if self._incld_loc_desc:
            loc_context.append(
                wrap_content(
                    clean_text(example_state['location_desc']),
                    consts.LOCATION_DESCRIPTION,
                )
            )
        return " ".join(loc_context)

    def surrounding_objects_context(self, example_state: Dict) -> str:
        out = ''
        if self._incld_surr_objs:
            content_str = consts.SET_MEMBERS_DELIM.join(
                example_state['surrounding_objs']
            )
            out = wrap_content(content_str, consts.SURROUNDING_OBJECTS)
        return out

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
        curr_state = example['state']
        return self.delim.join(
            [
                self.location_context(curr_state),
                self.surrounding_objects_context(curr_state),
            ]
        )

    def generate_example_label(self, example: Union[Dict, Message]) -> str:
        return knowledge_graph_as_str(example['state']['graph'])


class StateToActionTeacher(BaseJerichoWorldTeacher):
    """
    Game state to action.
    """

    def generate_example_text(self, example: Union[Dict, Message]) -> str:
        curr_state = example['state']
        return self.delim.join(
            [
                self.location_context(curr_state),
                self.surrounding_objects_context(curr_state),
            ]
        )

    def generate_example_label(self, example: Union[Dict, Message]) -> str:
        return example['action']


class DefaultTeacher(StateToKGTeacher):
    pass
