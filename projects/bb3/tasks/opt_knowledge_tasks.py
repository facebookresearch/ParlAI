#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List
from parlai.core.opt import Opt
import parlai.utils.logging as logging

from projects.bb3.tasks.r2c2_knowledge_tasks import (
    MsMarcoSearchKnowledgeTeacher as msmarco_skm,
    NQSearchKnowledgeTeacher as nq_skm,
    NQOpenSearchKnowledgeTeacher as nqopen_skm,
    NQOpenDialoguesSearchKnowledgeTeacher as nqopendialogues_skm,
    TriviaQASearchKnowledgeTeacher as triviaqa_skm,
    SquadSearchKnowledgeTeacher as squad_skm,
    WowSearchKnowledgeTeacher as wow_skm,
    WoiSearchKnowledgeTeacher as woi_skm,
    Convai2EntityKnowledgeTeacher as convai2_ckm,
    EDEntityKnowledgeTeacher as ed_ckm,
    MSCEntityKnowledgeTeacher as msc_ckm,
    BSTEntityKnowledgeTeacher as bst_ckm,
    Convai2MemoryKnowledgePersOverlapTeacher as convai2_pers_mkm,
    EDMemoryKnowledgePersOverlapTeacher as ed_pers_mkm,
    BSTMemoryKnowledgePersOverlapTeacher as bst_pers_mkm,
    MSCMemoryKnowledgePersOverlapTeacher as msc_pers_mkm,
    Convai2MemoryKnowledgeUttOverlapTeacher as convai2_utt_mkm,
    EDMemoryKnowledgeUttOverlapTeacher as ed_utt_mkm,
    BSTMemoryKnowledgeUttOverlapTeacher as bst_utt_mkm,
    MSCMemoryKnowledgeUttOverlapTeacher as msc_utt_mkm,
)

from projects.bb3.tasks.module_level_tasks import BB3TeacherMixin

from parlai.tasks.fits.agents import FitsBaseTeacher as FitsKnowledgeTeacher


def get_knowledge_opt_mutators(opt: Opt):
    mutators = '+'.join(
        ['prefix_speakers_opt', 'format_knowledge_tasks_for_decoder_only']
    )
    if opt.get('mutators'):
        mutators = '+'.join([opt['mutators'], mutators])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


class Convai2EntityKnowledgeTeacher(convai2_ckm):
    def get_special_mutators(self, opt) -> str:
        return '+'.join(
            [
                super().get_special_mutators(opt),
                'prefix_speakers_opt',
                'format_knowledge_tasks_for_decoder_only',
            ]
        )


class EDEntityKnowledgeTeacher(ed_ckm):
    def get_special_mutators(self, opt) -> str:
        return '+'.join(
            [
                super().get_special_mutators(opt),
                'prefix_speakers_opt',
                'format_knowledge_tasks_for_decoder_only',
            ]
        )


class MSCEntityKnowledgeTeacher(msc_ckm):
    def get_special_mutators(self, opt) -> str:
        return '+'.join(
            [
                super().get_special_mutators(opt),
                'prefix_speakers_opt',
                'format_knowledge_tasks_for_decoder_only',
            ]
        )


class BSTEntityKnowledgeTeacher(bst_ckm):
    def get_special_mutators(self, opt) -> str:
        return '+'.join(
            [
                super().get_special_mutators(opt),
                'prefix_speakers_opt',
                'format_knowledge_tasks_for_decoder_only',
            ]
        )


class Convai2MemoryKnowledgePersOverlapTeacher(convai2_pers_mkm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_knowledge_opt_mutators(opt)
        super().__init__(opt, shared)


class EDMemoryKnowledgePersOverlapTeacher(ed_pers_mkm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_knowledge_opt_mutators(opt)
        super().__init__(opt, shared)


class MSCMemoryKnowledgePersOverlapTeacher(msc_pers_mkm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_knowledge_opt_mutators(opt)
        super().__init__(opt, shared)


class BSTMemoryKnowledgePersOverlapTeacher(bst_pers_mkm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_knowledge_opt_mutators(opt)
        super().__init__(opt, shared)


class Convai2MemoryKnowledgeUttOverlapTeacher(convai2_utt_mkm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_knowledge_opt_mutators(opt)
        super().__init__(opt, shared)


class EDMemoryKnowledgeUttOverlapTeacher(ed_utt_mkm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_knowledge_opt_mutators(opt)
        super().__init__(opt, shared)


class MSCMemoryKnowledgeUttOverlapTeacher(msc_utt_mkm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_knowledge_opt_mutators(opt)
        super().__init__(opt, shared)


class BSTMemoryKnowledgeUttOverlapTeacher(bst_utt_mkm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_knowledge_opt_mutators(opt)
        super().__init__(opt, shared)


class MsMarcoSearchKnowledgeTeacher(msmarco_skm):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + [
            'prefix_speakers_opt',
            'format_knowledge_tasks_for_decoder_only_reduced_docs',
        ]


class NQSearchKnowledgeTeacher(nq_skm):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + [
            'prefix_speakers_opt',
            'format_knowledge_tasks_for_decoder_only_reduced_docs',
        ]


class NQOpenSearchKnowledgeTeacher(nqopen_skm):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + [
            'prefix_speakers_opt',
            'format_knowledge_tasks_for_decoder_only_reduced_docs',
        ]


class NQOpenDialoguesSearchKnowledgeTeacher(nqopendialogues_skm):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + [
            'prefix_speakers_opt',
            'format_knowledge_tasks_for_decoder_only_reduced_docs',
        ]


class TriviaQASearchKnowledgeTeacher(triviaqa_skm):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + [
            'prefix_speakers_opt',
            'format_knowledge_tasks_for_decoder_only_reduced_docs',
        ]


class SquadSearchKnowledgeTeacher(squad_skm):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + [
            'prefix_speakers_opt',
            'format_knowledge_tasks_for_decoder_only_reduced_docs',
        ]


class WowSearchKnowledgeTeacher(wow_skm):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + [
            'prefix_speakers_opt',
            'format_knowledge_tasks_for_decoder_only_reduced_docs',
        ]


class WoiSearchKnowledgeTeacher(woi_skm):
    def get_special_mutators(self) -> List[str]:
        return super().get_special_mutators() + [
            'prefix_speakers_opt',
            'format_knowledge_tasks_for_decoder_only_reduced_docs',
        ]


class FitsSearchKnowledgeTeacher(BB3TeacherMixin, FitsKnowledgeTeacher):
    def __init__(self, opt, shared=None):
        mutators = '+'.join(
            [
                'flatten_gold_knowledge_response_mutator',
                'fits_remove_special_toks_mutator',
                'add_selected_sentences_mutator',
                'prompt_knowledge_mutator',
                'prefix_speakers_opt',
                'format_knowledge_tasks_for_decoder_only_reduced_docs',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
