#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import List, Tuple

from parlai.core.opt import Opt
import parlai.tasks.convai2.agents as convai2
import parlai.tasks.blended_skill_talk.agents as bst
import parlai.tasks.empathetic_dialogues.agents as ed
import parlai.tasks.msc.agents as msc
import parlai.tasks.wizard_of_wikipedia.agents as wow
import parlai.utils.logging as logging

import projects.bb3.constants as BB3_CONST
from projects.bb3.tasks.module_level_tasks import BB3TeacherMixin

from projects.seeker.tasks.search_decision import (
    WoiSearchDecisionTeacher as woi_sdm,
    SquadSearchDecisionTeacher as squad_sdm,
    TriviaQASearchDecisionTeacher as triviaqa_sdm,
    NQSearchDecisionTeacher as nq_sdm,
    Convai2SearchDecisionTeacher as convai2_sdm,
    EDSearchDecisionTeacher as ed_sdm,
    MSCSearchDecisionTeacher as msc_sdm,
)


class WoiSearchDecisionTeacher(BB3TeacherMixin, woi_sdm):
    pass


class WowSearchDecisionTeacher(BB3TeacherMixin, wow.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'flatten',
                'wow_maybe_generate_search_query_mutator',
                'skip_retrieval_mutator',
                'filter_wow_topic_only_search_decision_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        opt['add_missing_turns'] = 'all'
        super().__init__(opt, shared)


class SquadSearchDecisionTeacher(BB3TeacherMixin, squad_sdm):
    pass


class TriviaQASearchDecisionTeacher(BB3TeacherMixin, triviaqa_sdm):
    pass


class NQOpenSearchDecisionTeacher(BB3TeacherMixin, nq_sdm):
    pass


class Convai2SearchDecisionTeacher(BB3TeacherMixin, convai2_sdm):
    pass


class EDSearchDecisionTeacher(BB3TeacherMixin, ed_sdm):
    pass


class MSCSearchDecisionTeacher(BB3TeacherMixin, msc_sdm):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'flatten',
                'filter_silence_only_mutator',
                'skip_retrieval_mutator',
                'bst_tasks_maybe_generate_search_query_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        opt['include_session1'] = False
        super().__init__(opt, shared)


def get_memory_decision_task_mutator(opt: Opt) -> str:
    """
    Set the mutators appropriately for the memory decision tasks.
    """
    mutators = '+'.join(
        [
            'memory_decision_mutator',
            'skip_retrieval_mutator',
            'prompt_memory_decision_mutator',
        ]
    )
    if opt.get('mutators'):
        mutators = '+'.join([mutators, opt['mutators']])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


def balance_data(episodes: List[Tuple]) -> List[Tuple]:
    """
    Balance the data so that we have equal numbers of memory/no memory.

    Very hacky and requires deep understanding of DialogData objects.

    Should in theory be used in a script to dump to disk.
    """
    random.seed(42)
    pos_inds = set()
    neg_inds = set()
    for i, ep in enumerate(episodes):
        # assume flattened
        assert len(ep) == 1
        ex = ep[0]
        label = ex[1][0] if isinstance(ex[1], list) else ex[1]
        if label == BB3_CONST.DO_ACCESS_MEMORY:
            pos_inds.add(i)
        elif label == BB3_CONST.DONT_ACCESS_MEMORY:
            neg_inds.add(i)
        else:
            raise ValueError(f"Incorrect label: {label}")
    pos_exs = [e for i, e in enumerate(episodes) if i in pos_inds]
    neg_exs = [e for i, e in enumerate(episodes) if i in neg_inds]
    new_data = []
    num_to_sample = min(len(pos_exs), len(neg_exs))
    if num_to_sample == len(pos_exs):
        new_data = pos_exs + random.sample(neg_exs, num_to_sample)
    else:
        new_data = neg_exs + random.sample(pos_exs, num_to_sample)

    return new_data


class Convai2MemoryDecisionTeacher(BB3TeacherMixin, convai2.NormalizedTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_decision_task_mutator(opt)
        opt['task'] += ':no_cands'
        super().__init__(opt, shared)


class EDMemoryDecisionTeacher(BB3TeacherMixin, ed.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_decision_task_mutator(opt)
        super().__init__(opt, shared)


class BSTMemoryDecisionTeacher(BB3TeacherMixin, bst.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_decision_task_mutator(opt)
        super().__init__(opt, shared)


class MSCMemoryDecisionTeacher(BB3TeacherMixin, msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'memory_decision_mutator',
                'skip_retrieval_mutator',
                'prompt_memory_decision_mutator',
                'filter_silence_only_memory_decision_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        opt['include_session1'] = False
        super().__init__(opt, shared)
