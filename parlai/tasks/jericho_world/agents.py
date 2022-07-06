#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from copy import deepcopy
from typing import Any, Dict, List, Set, Union, Optional, Tuple
import json
import logging
import os

from parlai.core.message import Message
from parlai.core.metrics import F1Metric
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


def graph_edge_as_str(subj: str, rel: str, obj: str) -> str:
    """
    Generate a formatted string from edge tuple components.
    """
    return '< ' + f' {consts.GRAPH_DELIM} '.join([subj, rel, obj]) + ' >'


def knowledge_graph_as_str(graph):
    if not graph:
        return consts.EMPTY_GRAPH_TOKEN

    graph_comps = []
    for edge_info in graph:
        # NOTE: often there are incomplete nodes that we detect and skip.
        # (Github issue reported here: https://github.com/JerichoWorld/JerichoWorld/issues/3)
        processed_edge = [e.strip() for e in edge_info if e.strip()]
        if len(processed_edge) == 3:
            s, r, o = processed_edge
            graph_comps.append(graph_edge_as_str(s, r, o))
    return (
        consts.SET_MEMBERS_DELIM.join(graph_comps)
        if graph_comps
        else consts.EMPTY_GRAPH_TOKEN
    )


def break_knowledge_graph(graph_str: str) -> Set[str]:
    return set([e.strip() for e in graph_str.split(consts.SET_MEMBERS_DELIM)])


def graph_mutation_diff(source_graph, dest_graph):
    """
    Generates the set of operations (ADD, DEL) needed to go from `source_graph` to
    `dest_graph`.
    """
    source_graph_edges = break_knowledge_graph(source_graph)
    dest_graph_edges = break_knowledge_graph(dest_graph)
    diff_edges = source_graph_edges.symmetric_difference(dest_graph_edges)

    mutations = []
    for edge in diff_edges:
        if edge in source_graph_edges:
            op = consts.GraphMutations.DEL
        else:
            op = consts.GraphMutations.ADD
        mutations.append(f'{op.name} {edge}')

    return mutations


def encode_set_elements(set1: Set[str], set2: Set[str]) -> Tuple[List[str]]:
    """
    Encodes (maps to int indices) elements in the union of two sets.
    """
    set1_enc = []
    set2_enc = []
    for el_id, el in enumerate(set1.union(set2)):
        el_id_str = str(el_id)
        if el in set1:
            set1_enc.append(el_id_str)
        if el in set2:
            set2_enc.append(el_id_str)
    return set1_enc, set2_enc


def extract_state_data(example_state: Dict, delim: str = ' ') -> Dict:
    def concat_vals(d):
        return [delim.join(p) for p in d.values()]

    return {
        'location_name': example_state['location']['name'],
        'observation': clean_text(example_state['obs']),
        'location_desc': clean_text(example_state['loc_desc']),
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
        self._incld_loc_name = opt['include_location']
        self.delim = opt['delimiter']
        self._incld_teacher_token = opt['include_teacher_token']
        self._incld_loc_desc = opt['include_location_description']
        self._incld_surr_objs = opt['include_surrounding_objects']
        self._prune_kg = opt['prune_knowledge_graph']
        self._keep_inv_during_kg_prune = False
        self.keep_next_state = True
        super().__init__(opt, shared=shared)

    def get_id(self):
        return 'JerichoWorldtBase'

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
            '--include-teacher-token',
            type='bool',
            default=True,
            help='Whether to include the name of the teacher',
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
            default=False,
            help='Whether to include the list of surrounding objects',
        )
        arg_group.add_argument(
            '--prune-knowledge-graph',
            type='bool',
            default=False,
            help='If true, items in knowledge graph that were not in the state description will be eliminiated.',
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

    def extract_knowledge_graph_str(
        self, state: Dict[str, Any], include_inv_objs: bool = False
    ) -> str:
        """
        Generates the string representing knowledge graph.

        It may modifiy the knowledge graph, for example when --prune-knowledge-graph is
        true.
        """

        def has_word_overlap(main_text, content_text_tokens):
            if not main_text or not content_text_tokens:
                return False

            for word in main_text.lower().split():
                if not word or len(word) < 3 or word in consts.GRAPH_VERT_SKIP_TOKEN:
                    continue
                if word in content_text_tokens:
                    return True
            return False

        def keep_edge(edge, text_tokens):
            """
            Returns false for edges that can NOT be readily inferred from the context.

            Edges that are referring to objects not referenced in the location
            description, except player (you). Thus, we drop these edges. Also, for some
            teachers (eg, ActionKGTeacher) the model can see user inventory before, and
            after the action, and they are important in result of action,. Thus, we keep
            < you , have, X > edges as well.
            """
            # Each graph edge is a tuple: (subject, relation, object)
            sub, rel, obj = [s.strip().lower() for s in edge]

            if sub == 'you':
                # player location
                if rel == 'in':
                    return True
                # user inventory object
                elif rel == 'have' and self._keep_inv_during_kg_prune:
                    return True

            return has_word_overlap(sub, text_tokens) and has_word_overlap(
                obj, text_tokens
            )

        if self._prune_kg:
            # Prunning the knowledge graph for the entities that are mentioned in the description
            old_graph = state['graph']
            new_graph = []

            # `text_desc` is the context that we show the model the generate the knowledge graph from.
            text_desc = f'{state["location_desc"]} {state["location_name"]}'
            if include_inv_objs:
                text_desc = f'{text_desc} {state["inventory_objs"]}'
            text_desc_tokens = text_desc.lower().split()

            # We prune the knowledge graph for items that are partially mentioned in `text_desc`
            for edge in old_graph:
                if keep_edge(edge, text_desc_tokens):
                    new_graph.append(edge)

            state['graph'] = new_graph

        return knowledge_graph_as_str(state['graph'])

    def generate_example_text_parts(self, example: Union[Dict, Message]) -> List[str]:
        """
        Returns the expected parts of the text for your example.

        These parts will be joined with `delim` to generate the text string.
        """
        teacher_token = (
            wrap_content(self.get_id(), 'tt') if self._incld_teacher_token else ''
        )

        curr_state = example['state']
        return [
            teacher_token,
            self.location_context(curr_state),
            self.surrounding_objects_context(curr_state),
        ]

    @abc.abstractmethod
    def generate_example_label(self, example: Union[Dict, Message]) -> str:
        """
        Implement this method for the expected label of your example.
        """

    def generate_example_label_candidates(
        self, example: Union[Dict, Message]
    ) -> List[str]:
        """
        Override this to include a list of label candidates with the example.
        """
        return []

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
        return ' '.join(loc_context)

    def surrounding_objects_context(self, example_state: Dict) -> str:
        out = ''
        if self._incld_surr_objs:
            content_str = consts.SET_MEMBERS_DELIM.join(
                example_state['surrounding_objs']
            )
            out = wrap_content(content_str, consts.SURROUNDING_OBJECTS)
        return out

    def knowledge_graph_context(self, example_state: Dict) -> str:
        """
        Generate the context for the flattened knowledge graph.
        """
        knowledge_graph = self.extract_knowledge_graph_str(example_state)
        return wrap_content(knowledge_graph, consts.KNOWLEDGE_GRAPH)

    def load_data(self, datafile):
        logging.info(f'Reading data from {datafile}')
        with open(datafile) as df:
            return json.load(df)

    def skip_example(self, example: Dict) -> bool:
        """
        Implements checks that some teachers have on valid examples, based on content.
        """
        return False

    def check_fix_state_kg(self, game: Dict, game_step_id: int) -> None:
        """
        Tries to fix the KG for the state associate with game_step_id, in place.

        It replaces the KG for game_step_id with the KG for `next_step` from previous
        step, or uses the `graph_diff` if game_step_id == 0.
        """
        if game[game_step_id]['state']['graph']:
            return

        if game_step_id == 0:
            # The first empty step, replace with the diff.
            game[0]['state']['graph'] = game[0]['graph_diff']
        else:
            # Get the graph from the next_state that was created in the previous step.
            prev_state_next = game[game_step_id - 1]['next_state']
            game[game_step_id]['state']['graph'] = deepcopy(prev_state_next['graph'])

    def _generate_example_text(self, text_parts: List[str]) -> str:
        """
        Concatnates the non-empty parts of text fetrue into a single string.
        """
        return self.delim.join([s.strip() for s in text_parts if s.strip()])

    def setup_data(self, datafile: str):
        for game in self.load_data(datafile):
            for step_i, game_step in enumerate(game):
                self.check_fix_state_kg(game, step_i)
                example = self._clean_example(game_step)
                if self.skip_example(example):
                    continue
                example['text'] = self._generate_example_text(
                    self.generate_example_text_parts(example)
                )

                # Preparing the label (received as str or list)
                label = self.generate_example_label(example)
                if not isinstance(label, list):
                    assert isinstance(label, str)
                    label = [label]
                example['labels'] = label

                label_candts = self.generate_example_label_candidates(example)
                if label_candts:
                    example['label_candidates'] = label_candts
                    if label not in label_candts:
                        # Making sure label is present in the candidates. Often the exact label
                        # may not be in the list of actions.
                        # This might be because of the short version or rephrase of it being used.
                        label_candts.extend(label)

                new_episode = step_i == 0
                yield example, new_episode


class BaseJerichoWorldSingleEpisodeTeacher(BaseJerichoWorldTeacher):
    """
    Truns each examples into a single episode: No history.
    """

    def setup_data(self, datafile: str):
        for example, _ in super().setup_data(datafile):
            yield example, True


###############################################################
#                                                             #
# Knowledge Graph and state prdiction teachers                #
#                                                             #
###############################################################


class StateToKGTeacher(BaseJerichoWorldSingleEpisodeTeacher):
    """
    Game state to the knowledge graph.
    """

    def get_id(self):
        return 'StateKG'

    def generate_example_label(self, example: Union[Dict, Message]) -> str:
        return self.extract_knowledge_graph_str(example['state'])

    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ) -> None:
        if model_response.is_padding() or (not model_response.get('text', None)):
            return

        expected_graph = break_knowledge_graph(labels[0].lower())
        predicted_graph = break_knowledge_graph(model_response['text'].lower())

        # Encoding the graph edges/mutation operations into ints for readily use of F1Metric
        expected_graph_enc, predicted_graph_enc = encode_set_elements(
            expected_graph, predicted_graph
        )
        self.metrics.add(
            'response_elements_f1',
            F1Metric.compute(
                guess=' '.join(predicted_graph_enc),
                answers=[' '.join(expected_graph_enc)],
            ),
        )

        # Subject, Relation F1
        # Changind "(MUT) < you , in , house >"   --into-->   "(MUT) < you , in "
        # This is to check F1 for the predicted subject and relation overlap.
        ekg_sub_rel = set([e.rsplit(',', 1)[0] for e in expected_graph])
        pkg_sub_rel = set([e.rsplit(',', 1)[0] for e in predicted_graph])
        ekg_sub_rel_ids, pkg_sub_rel_ids = encode_set_elements(ekg_sub_rel, pkg_sub_rel)
        self.metrics.add(
            'graph_subject_relation_f1',
            F1Metric.compute(
                guess=' '.join(pkg_sub_rel_ids), answers=[' '.join(ekg_sub_rel_ids)]
            ),
        )

        # Subject F1
        # Changind "(MUT) < you , in " (produced above)   --into-->   "(MUT) < you "
        # This is to check F1 for the predicted subject overlap.
        ekg_sub = set([e.split(',')[0] for e in ekg_sub_rel])
        pkg_sub = set([e.split(',')[0] for e in pkg_sub_rel])
        ekg_sub_ids, pkg_sub_ids = encode_set_elements(ekg_sub, pkg_sub)
        self.metrics.add(
            'graph_subject_f1',
            F1Metric.compute(
                guess=' '.join(pkg_sub_ids), answers=[' '.join(ekg_sub_ids)]
            ),
        )


class StaticKGTeacher(StateToKGTeacher):
    """
    Generates the knowledge graph from a single state.
    """

    def get_id(self):
        return 'StaticStateKG'

    def skip_example(self, example: Dict) -> bool:
        return (
            self.extract_knowledge_graph_str(example['state'])
            == consts.EMPTY_GRAPH_TOKEN
        )


class ActionKGTeacher(StateToKGTeacher):
    """
    Generates the knowledge graph mutations after a given action.
    """

    def __init__(self, opt: Opt, shared=None):
        self._keep_inv_during_kg_prune = True
        super().__init__(opt, shared=shared)

    def get_id(self):
        return 'Action2KGMutation'

    def skip_example(self, example: str):
        for st in ('state', 'next_state'):
            if (
                self.extract_knowledge_graph_str(example[st])
                == consts.EMPTY_GRAPH_TOKEN
            ):
                return True
        return False

    def generate_example_text_parts(self, example: Union[Dict, Message]) -> List[str]:
        prts = super().generate_example_text_parts(example)

        # Adding knowledge graph to the text.
        prts.append(self.knowledge_graph_context(example['state']))

        # Adding observation after the action to the text.
        prts.append(
            wrap_content(example['next_state']['observation'], consts.OBSERVATION)
        )

        prts.append(wrap_content(example['action'], consts.ACTION))
        return prts

    def generate_example_label(self, example: Union[Dict, Message]) -> str:
        curr_graph = self.extract_knowledge_graph_str(
            example['state'], include_inv_objs=True
        )
        next_graph = self.extract_knowledge_graph_str(
            example['next_state'], include_inv_objs=True
        )
        graph_diff = graph_mutation_diff(curr_graph, next_graph)
        return (
            # sorting to pass the tests, otherwise the results are in various orders.
            '\n'.join(sorted(graph_diff))
            if graph_diff
            else consts.GraphMutations.NO_MUTATION.name
        )


###############################################################
#                                                             #
# Action prediction teachers                                  #
#                                                             #
###############################################################


class ActionPredictionBase(BaseJerichoWorldSingleEpisodeTeacher):
    """
    The base class for predicting the actions (or set of valid action).

    This is an abstract class, use its children for predicting actions.
    """

    def __init__(self, opt: Opt, shared=None):
        self._incld_knwldg_grph = opt['include_knowledge_graph']
        self._keep_inv_during_kg_prune = True
        super().__init__(opt, shared=shared)

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt=None) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        arg_group = parser.add_argument_group('Jericho World Action Teacher Args')
        arg_group.add_argument(
            '--include-knowledge-graph',
            type='bool',
            default=True,
            help='Whether to include the knowledge graph.',
        )

    def generate_example_text_parts(self, example: Union[Dict, Message]) -> List[str]:
        prts = super().generate_example_text_parts(example)
        if self._incld_knwldg_grph:
            prts.append(self.knowledge_graph_context(example['state']))
        return prts


class StateToValidActionsTeacher(ActionPredictionBase):
    """
    Generates the set of valid actions.
    """

    def get_id(self):
        return 'ValidActionsTeacher'

    def generate_example_label(self, example: Union[Dict, Message]) -> List[str]:
        return example['state']['valid_acts']


class StateToActionTeacher(ActionPredictionBase):
    """
    Game state to action (selecting between the set of valid actions).
    """

    def get_id(self):
        return 'NextActionTeacher'

    def generate_example_label_candidates(
        self, example: Union[Dict, Message]
    ) -> List[str]:
        """
        Override this to include a list of label candidates with the example.
        """
        return example['state']['valid_acts']

    def generate_example_label(self, example: Union[Dict, Message]) -> str:
        return example['action']


class DefaultTeacher(StaticKGTeacher):
    pass
