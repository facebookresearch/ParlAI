#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
SeeKeR Dialogue Tasks.
"""
from typing import Optional

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import MultiTaskTeacher
import parlai.tasks.convai2.agents as convai2
import parlai.tasks.blended_skill_talk.agents as bst
import parlai.tasks.empathetic_dialogues.agents as ed
import parlai.tasks.wizard_of_internet.agents as woi
import parlai.tasks.wizard_of_wikipedia.agents as wow
import parlai.tasks.msc.agents as msc
import parlai.tasks.ms_marco.agents as ms_marco

import parlai.utils.logging as logging

import projects.seeker.tasks.mutators  # type: ignore   # noqa: F401


class WoiDialogueTeacher(woi.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'flatten',
                'woi_pop_documents_mutator',
                'woi_filter_no_passage_used',
                'woi_add_checked_sentence_to_input',
                'skip_retrieval_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = "WoiDialogueTeacher"


class WowDialogueTeacher(wow.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['add_missing_turns'] = 'all'
        mutators = '+'.join(
            [
                'flatten',
                'wow_filter_no_passage_used',
                'wow_add_checked_sentence_to_input',
                'skip_retrieval_mutator',
                'wow_to_woi',
                'woi_pop_documents_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = "WowDialogueTeacher"


class MsMarcoDialogueTeacher(ms_marco.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'ms_marco_filter_has_answer',
                'ms_marco_create_fid_docs',
                'ms_marco_find_selected_sentence_for_response',
                'woi_pop_documents_mutator',
                'skip_retrieval_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = "MsMarcoDialogueTeacher"


def get_dialogue_task_mutators(opt: Opt) -> str:
    """
    Set the mutators appropriately for the dialogue tasks.
    """
    mutators = '+'.join(
        ['flatten', 'extract_entity_for_response_model', 'skip_retrieval_mutator']
    )
    if opt.get('mutators'):
        mutators = '+'.join([mutators, opt['mutators']])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


class Convai2DialogueTeacher(convai2.NormalizedTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_dialogue_task_mutators(opt)
        opt['task'] += ':no_cands'
        super().__init__(opt, shared)
        self.id = 'Convai2DialogueTeacher'


class EDDialogueTeacher(ed.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_dialogue_task_mutators(opt)
        super().__init__(opt, shared)
        self.id = 'EDDialogueTeacher'


class BSTDialogueTeacher(bst.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_dialogue_task_mutators(opt)
        super().__init__(opt, shared)
        self.id = 'BSTDialogueTeacher'


class MSCDialogueTeacher(msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_dialogue_task_mutators(opt)
        opt['include_session1'] = False
        super().__init__(opt, shared)
        self.id = 'MSCDialogueTeacher'


class MSCDialogueOverlapTeacher(msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = '+'.join(
            ['flatten', 'msc_find_selected_sentence_response', 'skip_retrieval_mutator']
        )
        opt['include_session1'] = False
        super().__init__(opt, shared)
        self.id = 'MSCDialogueOverlapTeacher'


class DialogueTeacher(MultiTaskTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        WoiDialogueTeacher.add_cmdline_args(parser, partial_opt)
        WowDialogueTeacher.add_cmdline_args(parser, partial_opt)
        MsMarcoDialogueTeacher.add_cmdline_args(parser, partial_opt)
        Convai2DialogueTeacher.add_cmdline_args(parser, partial_opt)
        EDDialogueTeacher.add_cmdline_args(parser, partial_opt)
        BSTDialogueTeacher.add_cmdline_args(parser, partial_opt)
        MSCDialogueTeacher.add_cmdline_args(parser, partial_opt)
        MSCDialogueOverlapTeacher.add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt, shared=None):
        tasks = [
            f"projects.seeker.tasks.dialogue:{teacher}"
            for teacher in [
                'WoiDialogueTeacher',
                'WowDialogueTeacher',
                'MsMarcoDialogueTeacher',
                'Convai2DialogueTeacher',
                'EDDialogueTeacher',
                'BSTDialogueTeacher',
                'MSCDialogueTeacher',
                'MSCDialogueOverlapTeacher',
            ]
        ]
        opt['task'] = ','.join(tasks)
        super().__init__(opt, shared)


class DefaultTeacher(DialogueTeacher):
    pass
