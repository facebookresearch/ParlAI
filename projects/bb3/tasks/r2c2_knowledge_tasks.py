#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Knowledge tasks used to train the BB3-3B Model.
"""
from typing import List
from parlai.core.opt import Opt
import parlai.tasks.convai2.agents as convai2
import parlai.tasks.blended_skill_talk.agents as bst
import parlai.tasks.empathetic_dialogues.agents as ed
import parlai.tasks.msc.agents as msc
import parlai.utils.logging as logging

from projects.seeker.tasks.knowledge import (
    WoiKnowledgeTeacher,
    WowKnowledgeTeacher,
    MsMarcoKnowledgeTeacher,
    SquadKnowledgeTeacher,
    TriviaQAKnowledgeTeacher,
    NQKnowledgeTeacher,
    NQOpenKnowledgeTeacher,
    NQOpenDialoguesKnowledgeTeacher,
    Convai2KnowledgeTeacher,
    EDKnowledgeTeacher,
    BSTKnowledgeTeacher,
    MSCKnowledgeTeacher,
)

from projects.bb3.tasks.module_level_tasks import BB3TeacherMixin

from parlai.tasks.fits.agents import FitsBaseTeacher as FitsKnowledgeTeacher

####################
# Search Knowledge #
####################


class WoiSearchKnowledgeTeacher(BB3TeacherMixin, WoiKnowledgeTeacher):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + ['prompt_knowledge_mutator']


class WowSearchKnowledgeTeacher(BB3TeacherMixin, WowKnowledgeTeacher):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + ['prompt_knowledge_mutator']


class MsMarcoSearchKnowledgeTeacher(BB3TeacherMixin, MsMarcoKnowledgeTeacher):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + ['prompt_knowledge_mutator']


class SquadSearchKnowledgeTeacher(BB3TeacherMixin, SquadKnowledgeTeacher):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + ['prompt_knowledge_mutator']


class TriviaQASearchKnowledgeTeacher(BB3TeacherMixin, TriviaQAKnowledgeTeacher):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + ['prompt_knowledge_mutator']


class NQSearchKnowledgeTeacher(BB3TeacherMixin, NQKnowledgeTeacher):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + ['prompt_knowledge_mutator']


class NQOpenSearchKnowledgeTeacher(BB3TeacherMixin, NQOpenKnowledgeTeacher):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + ['prompt_knowledge_mutator']


class NQOpenDialoguesSearchKnowledgeTeacher(
    BB3TeacherMixin, NQOpenDialoguesKnowledgeTeacher
):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + ['prompt_knowledge_mutator']


class Convai2EntityKnowledgeTeacher(BB3TeacherMixin, Convai2KnowledgeTeacher):
    def get_special_mutators(self, opt) -> str:
        return '+'.join(
            [super().get_special_mutators(opt), 'prompt_extract_entity_mutator']
        )


class EDEntityKnowledgeTeacher(BB3TeacherMixin, EDKnowledgeTeacher):
    def get_special_mutators(self, opt) -> str:
        return '+'.join(
            [super().get_special_mutators(opt), 'prompt_extract_entity_mutator']
        )


class BSTEntityKnowledgeTeacher(BB3TeacherMixin, BSTKnowledgeTeacher):
    def get_special_mutators(self, opt) -> str:
        return '+'.join(
            [super().get_special_mutators(opt), 'prompt_extract_entity_mutator']
        )


class MSCEntityKnowledgeTeacher(BB3TeacherMixin, MSCKnowledgeTeacher):
    def get_special_mutators(self, opt) -> str:
        return '+'.join(
            [super().get_special_mutators(opt), 'prompt_extract_entity_mutator']
        )


class FitsSearchKnowledgeTeacher(BB3TeacherMixin, FitsKnowledgeTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'flatten_gold_knowledge_response_mutator',
                'fits_remove_special_toks_mutator',
                'add_selected_sentences_mutator',
                'prompt_knowledge_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)


###########################################
# Memory Knowledge from utterance overlap #
###########################################


def get_memory_knowledge_utt_overlap_task_mutator(opt: Opt) -> str:
    """
    Set the mutators appropriately for the dialogue tasks.
    """
    mutators = '+'.join(
        [
            'flatten',
            'filter_silence_only_mutator',
            'msc_find_selected_sentence_knowledge',
            'add_retrieved_documents_mutator',
            'skip_retrieval_mutator',
            'prompt_knowledge_mutator',
            'convert_overlap_to_personas_as_docs',
            'ensure_same_number_docs_and_titles_mutator',
            'fix_mkm_formatting_mutator',
            'normalize_reply_mutator',
        ]
    )
    if opt.get('mutators'):
        mutators = '+'.join([mutators, opt['mutators']])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


class Convai2MemoryKnowledgeUttOverlapTeacher(
    BB3TeacherMixin, convai2.NormalizedTeacher
):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_utt_overlap_task_mutator(opt)
        opt['task'] += ':no_cands'
        super().__init__(opt, shared)


class EDMemoryKnowledgeUttOverlapTeacher(BB3TeacherMixin, ed.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_utt_overlap_task_mutator(opt)
        super().__init__(opt, shared)


class BSTMemoryKnowledgeUttOverlapTeacher(BB3TeacherMixin, bst.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_utt_overlap_task_mutator(opt)
        super().__init__(opt, shared)


class MSCMemoryKnowledgeUttOverlapTeacher(BB3TeacherMixin, msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_utt_overlap_task_mutator(opt)
        opt['include_session1'] = False
        super().__init__(opt, shared)


##################################################
# Memory Knowledge from Computed Persona Overlap #
##################################################


def get_memory_knowledge_pers_overlap_task_mutator(opt: Opt) -> str:
    """
    Set the mutators appropriately for the dialogue tasks.
    """
    mutators = '+'.join(
        [
            'personas_as_docs',
            'filter_silence_only_mutator',
            'match_target_to_persona_mutator',
            'prompt_access_memory_mutator',
            'normalize_reply_mutator',
            'ensure_same_number_docs_and_titles_mutator',
        ]
    )
    if opt.get('mutators'):
        mutators = '+'.join([mutators, opt['mutators']])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


class Convai2MemoryKnowledgePersOverlapTeacher(
    BB3TeacherMixin, convai2.NormalizedTeacher
):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_pers_overlap_task_mutator(opt)
        opt['task'] += ':no_cands'
        super().__init__(opt, shared)


class EDMemoryKnowledgePersOverlapTeacher(BB3TeacherMixin, ed.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_pers_overlap_task_mutator(opt)
        super().__init__(opt, shared)


class BSTMemoryKnowledgePersOverlapTeacher(BB3TeacherMixin, bst.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_pers_overlap_task_mutator(opt)
        super().__init__(opt, shared)


class MSCMemoryKnowledgePersOverlapTeacher(BB3TeacherMixin, msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_pers_overlap_task_mutator(opt)
        opt['include_session1'] = False
        super().__init__(opt, shared)
