#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
SeeKeR Search Decision Tasks.
"""
from typing import Optional

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import MultiTaskTeacher
import parlai.tasks.convai2.agents as convai2
import parlai.tasks.empathetic_dialogues.agents as ed
import parlai.tasks.wizard_of_internet.agents as woi
import parlai.tasks.wizard_of_wikipedia.agents as wow
import parlai.tasks.squad.agents as squad
import parlai.tasks.triviaqa.agents as triviaqa
import parlai.tasks.natural_questions.agents as nq
import parlai.tasks.msc.agents as msc

import parlai.utils.logging as logging

import projects.seeker.tasks.mutators  # type: ignore


class WoiSearchDecisionTeacher(woi.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'flatten',
                'woi_dropout_retrieved_docs',
                'woi_maybe_generate_search_query_mutator',
                'woi_pop_documents_mutator',
                'skip_retrieval_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = 'WoiSearchDecisionTeacher'


class WowSearchDecisionTeacher(wow.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'flatten',
                'wow_maybe_generate_search_query_mutator',
                'skip_retrieval_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        opt['add_missing_turns'] = 'all'
        super().__init__(opt, shared)
        self.id = 'WowSearchDecisionTeacher'


class SquadSearchDecisionTeacher(squad.OpensquadTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            ['do_generate_search_query_mutator', 'skip_retrieval_mutator']
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = 'SquadSearchDecisionTeacher'


class TriviaQASearchDecisionTeacher(triviaqa.NoEvidenceWebTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            ['do_generate_search_query_mutator', 'skip_retrieval_mutator']
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = 'TriviaQASearchDecisionTeacher'


class NQSearchDecisionTeacher(nq.NaturalQuestionsOpenTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            ['do_generate_search_query_mutator', 'skip_retrieval_mutator']
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = 'NQSearchDecisionTeacher'


def get_dialogue_task_mutators(opt: Opt) -> str:
    """
    Set the mutators appropriately for the dialogue tasks.
    """
    mutators = '+'.join(
        [
            'flatten',
            'skip_retrieval_mutator',
            'bst_tasks_maybe_generate_search_query_mutator',
        ]
    )
    if opt.get('mutators'):
        mutators = '+'.join([mutators, opt['mutators']])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


class Convai2SearchDecisionTeacher(convai2.NormalizedTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_dialogue_task_mutators(opt)
        opt['task'] += ':no_cands'
        super().__init__(opt, shared)
        self.id = 'Convai2SearchDecisionTeacher'


class EDSearchDecisionTeacher(ed.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_dialogue_task_mutators(opt)
        super().__init__(opt, shared)
        self.id = 'EDSearchDecisionTeacher'


class MSCSearchDecisionTeacher(msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_dialogue_task_mutators(opt)
        opt['include_session1'] = False
        super().__init__(opt, shared)
        self.id = 'MSCSearchDecisionTeacher'


class SearchDecisionTeacher(MultiTaskTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        WoiSearchDecisionTeacher.add_cmdline_args(parser, partial_opt)
        WowSearchDecisionTeacher.add_cmdline_args(parser, partial_opt)
        SquadSearchDecisionTeacher.add_cmdline_args(parser, partial_opt)
        TriviaQASearchDecisionTeacher.add_cmdline_args(parser, partial_opt)
        NQSearchDecisionTeacher.add_cmdline_args(parser, partial_opt)
        Convai2SearchDecisionTeacher.add_cmdline_args(parser, partial_opt)
        EDSearchDecisionTeacher.add_cmdline_args(parser, partial_opt)
        MSCSearchDecisionTeacher.add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        tasks = [
            f"projects.seeker.tasks.search_decision:{teacher}"
            for teacher in [
                'WoiSearchDecisionTeacher',
                'WowSearchDecisionTeacher',
                'SquadSearchDecisionTeacher',
                'TriviaQASearchDecisionTeacher',
                'NQSearchDecisionTeacher',
                'Convai2SearchDecisionTeacher',
                'EDSearchDecisionTeacher',
                'MSCSearchDecisionTeacher',
            ]
        ]
        opt['task'] = ','.join(tasks)
        super().__init__(opt, shared)


class DefaultTeacher(SearchDecisionTeacher):
    pass
