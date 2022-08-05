#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.opt import Opt
import parlai.utils.logging as logging
import parlai.tasks.convai2.agents as convai2
import parlai.tasks.blended_skill_talk.agents as bst
import parlai.tasks.empathetic_dialogues.agents as ed
import parlai.tasks.msc.agents as msc
import parlai.tasks.funpedia.agents as funpedia
import parlai.tasks.google_sgd.agents as google_sgd
import parlai.tasks.taskmaster.agents as taskmaster
import parlai.tasks.taskmaster2.agents as taskmaster2
import parlai.tasks.taskmaster3.agents as taskmaster3
import parlai.tasks.saferdialogues.agents as Saferdialogues
import parlai.tasks.wizard_of_wikipedia.agents as wow
import parlai.tasks.wizard_of_internet.agents as woi
import parlai.tasks.light_dialog.agents as light
import parlai.tasks.light_dialog_wild.agents as light_wild
import parlai.tasks.style_gen.agents as style_gen


from projects.seeker.tasks.dialogue import (
    WoiDialogueTeacher as woi_srm,
    WowDialogueTeacher as wow_srm,
    MsMarcoDialogueTeacher as msmarco_srm,
)

from parlai.tasks.fits.agents import FitsBaseTeacher as FitsDialogueTeacher
import parlai.tasks.fits.mutators  # type: ignore

from projects.bb3.tasks.module_level_tasks import BB3TeacherMixin
from projects.bb3.tasks.r2c2_knowledge_tasks import (
    get_memory_knowledge_utt_overlap_task_mutator,
    get_memory_knowledge_pers_overlap_task_mutator,
)

##################
# SearchDialogue #
##################


class WowSearchDialogueTeacher(BB3TeacherMixin, wow_srm):
    pass


class WoiSearchDialogueTeacher(BB3TeacherMixin, woi_srm):
    pass


class MsMarcoSearchDialogueTeacher(BB3TeacherMixin, msmarco_srm):
    pass


class FitsSearchDialogueTeacher(BB3TeacherMixin, FitsDialogueTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'flatten_gold_human_mutator',
                'knowledge_appended_mutator',
                'skip_retrieval_mutator',
                'no_woi_gold_docs_mutator',
                'fits_pop_keys_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)


class FunpediaSearchDialogueTeacher(BB3TeacherMixin, funpedia.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(['funpedia_to_bb3_mutator', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)


class GoogleSgdSearchDialogueTeacher(BB3TeacherMixin, google_sgd.SystemTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(['tod_to_srm_mutator', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)


class TaskmasterSearchDialogueTeacher(BB3TeacherMixin, taskmaster.SystemTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(['tod_to_srm_mutator', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)


class Taskmaster2SearchDialogueTeacher(BB3TeacherMixin, taskmaster2.SystemTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(['tod_to_srm_mutator', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)


class Taskmaster3SearchDialogueTeacher(BB3TeacherMixin, taskmaster3.SystemTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(['tod_to_srm_mutator', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)


###################
# Entity Dialogue #
###################


def get_entity_dialogue_task_mutators(opt: Opt) -> str:
    """
    Set the mutators appropriately for the dialogue tasks.
    """
    mutators = '+'.join(
        ['flatten', 'extract_entity_for_response_model_bb3', 'skip_retrieval_mutator']
    )
    if opt.get('mutators'):
        mutators = '+'.join([mutators, opt['mutators']])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


class Convai2EntityDialogueTeacher(BB3TeacherMixin, convai2.NormalizedTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_entity_dialogue_task_mutators(opt)
        opt['task'] += ':no_cands'
        super().__init__(opt, shared)


class EDEntityDialogueTeacher(BB3TeacherMixin, ed.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_entity_dialogue_task_mutators(opt)
        super().__init__(opt, shared)


class BSTEntityDialogueTeacher(BB3TeacherMixin, bst.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_entity_dialogue_task_mutators(opt)
        super().__init__(opt, shared)


class MSCEntityDialogueTeacher(BB3TeacherMixin, msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_entity_dialogue_task_mutators(opt)
        opt['include_session1'] = False
        super().__init__(opt, shared)


###################
# Memory Dialogue #
###################


def get_memory_dialogue_task_mutator(opt: Opt) -> str:
    """
    Set the mutators appropriately for the dialogue tasks.
    """
    mutators = '+'.join(
        [
            'convert_mkm_to_mrm_mutator',
            'skip_retrieval_mutator',
        ]
    )
    assert opt.get('mutators')
    mutators = '+'.join([opt['mutators'], mutators])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


class Convai2MemoryDialogueFromPersOverlapTeacher(
    BB3TeacherMixin, convai2.NormalizedTeacher
):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_pers_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        opt['task'] += ':no_cands'
        super().__init__(opt, shared)


class EDMemoryDialogueFromPersOverlapTeacher(BB3TeacherMixin, ed.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_pers_overlap_task_mutator(opt)
        mutators = get_memory_dialogue_task_mutator(opt)
        opt['mutators'] = mutators
        super().__init__(opt, shared)


class BSTMemoryDialogueFromPersOverlapTeacher(BB3TeacherMixin, bst.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_pers_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        super().__init__(opt, shared)


class MSCMemoryDialogueFromPersOverlapTeacher(BB3TeacherMixin, msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_pers_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        opt['include_session1'] = False
        super().__init__(opt, shared)


class Convai2MemoryDialogueFromUttOverlapTeacher(
    BB3TeacherMixin, convai2.NormalizedTeacher
):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_utt_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        opt['task'] += ':no_cands'
        super().__init__(opt, shared)


class EDMemoryDialogueFromUttOverlapTeacher(BB3TeacherMixin, ed.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_utt_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        super().__init__(opt, shared)


class BSTMemoryDialogueFromUttOverlapTeacher(BB3TeacherMixin, bst.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_utt_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        super().__init__(opt, shared)


class MSCMemoryDialogueFromUttOverlapTeacher(BB3TeacherMixin, msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_utt_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        opt['include_session1'] = False
        super().__init__(opt, shared)


####################
# Vanilla Dialogue #
####################


class SaferdialoguesVanillaDialogueTeacher(
    BB3TeacherMixin, Saferdialogues.DefaultTeacher
):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(['flatten', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)


class WowVanillaDialogueTeacher(BB3TeacherMixin, wow.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'flatten',
                'wow_only_no_passage_used',
                'wow_to_woi',
                'woi_pop_documents_mutator',
                'skip_retrieval_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        opt['add_missing_turns'] = 'all'
        super().__init__(opt, shared)


class WoiVanillaDialogueTeacher(BB3TeacherMixin, woi.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'flatten',
                'woi_keep_only_no_passage_used',
                'skip_retrieval_mutator',
                'woi_pop_documents_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)


class LightVanillaDialogueTeacher(BB3TeacherMixin, light.SimpleMultiTeacher):
    def __init__(self, opt, shared=None):
        opt['light_label'] = 'speech'
        mutators = '+'.join(['flatten', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)


class LightWildVanillaDialogueTeacher(BB3TeacherMixin, light_wild.SimpleMultiTeacher):
    def __init__(self, opt, shared=None):
        opt['light_label'] = 'speech'
        mutators = '+'.join(['flatten', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)


class EDVanillaDialogueTeacher(BB3TeacherMixin, ed.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(['flatten', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)


class Convai2VanillaDialogueTeacher(BB3TeacherMixin, convai2.NormalizedTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(['flatten', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        opt['task'] += ':no_cands'
        super().__init__(opt, shared)


class MSCVanillaDialogueTeacher(BB3TeacherMixin, msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(['flatten', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        opt['include_session1'] = False
        super().__init__(opt, shared)


#####################
# Grounded Dialogue #
#####################


class Convai2StyleGroundingDialogueTeacher(
    BB3TeacherMixin, style_gen.LabeledConvAI2PersonaTopicifierTeacher
):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(['flatten', 'style_gen_to_grm', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        opt['include_session1'] = False
        super().__init__(opt, shared)


class BSTStyleGroundingDialogueTeacher(
    BB3TeacherMixin, style_gen.LabeledBlendedSkillTalkTeacher
):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(['flatten', 'style_gen_to_grm', 'skip_retrieval_mutator'])
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        opt['include_session1'] = False
        super().__init__(opt, shared)
