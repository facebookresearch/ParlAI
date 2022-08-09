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

from projects.bb3.tasks.r2c2_dialogue_tasks import (
    WoiSearchDialogueTeacher as woi_srm,
    WowSearchDialogueTeacher as wow_srm,
    MsMarcoSearchDialogueTeacher as msmarco_srm,
    FitsSearchDialogueTeacher as fits_srm,
    FunpediaSearchDialogueTeacher as funpedia_srm,
    GoogleSgdSearchDialogueTeacher as googlesgd_srm,
    TaskmasterSearchDialogueTeacher as taskmaster_srm,
    Taskmaster2SearchDialogueTeacher as taskmaster2_srm,
    Taskmaster3SearchDialogueTeacher as taskmaster3_srm,
    Convai2EntityDialogueTeacher as convai2_crm,
    EDEntityDialogueTeacher as ed_crm,
    BSTEntityDialogueTeacher as bst_crm,
    MSCEntityDialogueTeacher as msc_crm,
    SaferdialoguesVanillaDialogueTeacher as Saferdialogues_vrm,
    WoiVanillaDialogueTeacher as woi_vrm,
    WowVanillaDialogueTeacher as wow_vrm,
    LightVanillaDialogueTeacher as light_vrm,
    LightWildVanillaDialogueTeacher as lightwild_vrm,
    EDVanillaDialogueTeacher as ed_vrm,
    Convai2VanillaDialogueTeacher as convai2_vrm,
    MSCVanillaDialogueTeacher as msc_vrm,
    Convai2StyleGroundingDialogueTeacher as convai2_grm,
    BSTStyleGroundingDialogueTeacher as bst_grm,
    get_memory_knowledge_pers_overlap_task_mutator,
    get_memory_knowledge_utt_overlap_task_mutator,
    get_memory_dialogue_task_mutator,
)

from projects.bb3.tasks.module_level_tasks import BB3TeacherMixin


def get_prefix_speakers_opt_mutators(opt: Opt):
    mutators = '+'.join(['prefix_speakers_opt'])
    if opt.get('mutators'):
        mutators = '+'.join([opt['mutators'], mutators])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


def get_dialogue_opt_mutators(opt: Opt):
    mutators = '+'.join(
        ['prefix_speakers_opt', 'format_response_tasks_for_decoder_only']
    )
    if opt.get('mutators'):
        mutators = '+'.join([opt['mutators'], mutators])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


def get_vanilla_dialogue_opt_mutators(opt: Opt):
    mutators = '+'.join(
        ['prefix_speakers_opt', 'format_vanilla_dialogue_for_decoder_only']
    )
    if opt.get('mutators'):
        mutators = '+'.join([opt['mutators'], mutators])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


def get_light_dialogue_opt_mutators(opt: Opt):
    mutators = '+'.join(['prefix_speakers_opt', 'format_light_tasks_for_decoder_only'])
    if opt.get('mutators'):
        mutators = '+'.join([opt['mutators'], mutators])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


def get_style_dialogue_opt_mutators(opt: Opt):
    mutators = '+'.join(
        ['prefix_speakers_opt', 'format_style_grounding_tasks_for_decoder_only']
    )
    if opt.get('mutators'):
        mutators = '+'.join([opt['mutators'], mutators])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


class Convai2EntityDialogueTeacher(convai2_crm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class EDEntityDialogueTeacher(ed_crm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class MSCEntityDialogueTeacher(msc_crm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class BSTEntityDialogueTeacher(bst_crm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class Convai2MemoryDialogueFromPersOverlapTeacher(
    BB3TeacherMixin, convai2.NormalizedTeacher
):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_pers_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        opt['task'] += ':no_cands'
        super().__init__(opt, shared)


class EDMemoryDialogueFromPersOverlapTeacher(BB3TeacherMixin, ed.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_pers_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class BSTMemoryDialogueFromPersOverlapTeacher(BB3TeacherMixin, bst.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_pers_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class MSCMemoryDialogueFromPersOverlapTeacher(BB3TeacherMixin, msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_pers_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        opt['include_session1'] = False
        super().__init__(opt, shared)


class Convai2MemoryDialogueFromUttOverlapTeacher(
    BB3TeacherMixin, convai2.NormalizedTeacher
):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_utt_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        opt['task'] += ':no_cands'
        super().__init__(opt, shared)


class EDMemoryDialogueFromUttOverlapTeacher(BB3TeacherMixin, ed.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_utt_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class BSTMemoryDialogueFromUttOverlapTeacher(BB3TeacherMixin, bst.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_utt_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class MSCMemoryDialogueFromUttOverlapTeacher(BB3TeacherMixin, msc.DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt['mutators'] = get_memory_knowledge_utt_overlap_task_mutator(opt)
        opt['mutators'] = get_memory_dialogue_task_mutator(opt)
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        opt['include_session1'] = False
        super().__init__(opt, shared)


class MsMarcoSearchDialogueTeacher(msmarco_srm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class WowSearchDialogueTeacher(wow_srm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class WoiSearchDialogueTeacher(woi_srm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class FitsSearchDialogueTeacher(fits_srm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class FunpediaSearchDialogueTeacher(funpedia_srm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class GoogleSgdSearchDialogueTeacher(googlesgd_srm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class TaskmasterSearchDialogueTeacher(taskmaster_srm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class Taskmaster2SearchDialogueTeacher(taskmaster2_srm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class Taskmaster3SearchDialogueTeacher(taskmaster3_srm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class WowVanillaDialogueTeacher(wow_vrm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_prefix_speakers_opt_mutators(opt)
        super().__init__(opt, shared)


class WoiVanillaDialogueTeacher(woi_vrm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_prefix_speakers_opt_mutators(opt)
        super().__init__(opt, shared)


class EDVanillaDialogueTeacher(ed_vrm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_prefix_speakers_opt_mutators(opt)
        super().__init__(opt, shared)


class SaferdialoguesVanillaDialogueTeacher(Saferdialogues_vrm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_prefix_speakers_opt_mutators(opt)
        super().__init__(opt, shared)


class Convai2VanillaDialogueTeacher(convai2_vrm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_vanilla_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class MSCVanillaDialogueTeacher(msc_vrm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_vanilla_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class LightVanillaDialogueTeacher(light_vrm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_light_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class LightWildVanillaDialogueTeacher(lightwild_vrm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_light_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class Convai2StyleGroundingDialogueTeacher(convai2_grm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_style_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)


class BSTStyleGroundingDialogueTeacher(bst_grm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_style_dialogue_opt_mutators(opt)
        super().__init__(opt, shared)
