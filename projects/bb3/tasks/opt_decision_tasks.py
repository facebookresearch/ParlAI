#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.opt import Opt
import parlai.utils.logging as logging

from projects.bb3.tasks.r2c2_decision_tasks import (
    Convai2MemoryDecisionTeacher as convai2_mdm,
    EDMemoryDecisionTeacher as ed_mdm,
    BSTMemoryDecisionTeacher as bst_mdm,
    MSCMemoryDecisionTeacher as msc_mdm,
    WowSearchDecisionTeacher as wow_sdm,
    NQOpenSearchDecisionTeacher as nq_sdm,
    SquadSearchDecisionTeacher as squad_sdm,
    TriviaQASearchDecisionTeacher as tqa_sdm,
    Convai2SearchDecisionTeacher as convai2_sdm,
    EDSearchDecisionTeacher as ed_sdm,
    MSCSearchDecisionTeacher as msc_sdm,
    WoiSearchDecisionTeacher as woi_sdm,
)


def get_decision_opt_mutators(opt: Opt):
    mutators = '+'.join(
        ['prefix_speakers_opt', 'format_decision_tasks_for_decoder_only']
    )
    if opt.get('mutators'):
        mutators = '+'.join([opt['mutators'], mutators])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


class NQOpenSearchDecisionTeacher(nq_sdm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_decision_opt_mutators(opt)
        super().__init__(opt, shared)


class SquadSearchDecisionTeacher(squad_sdm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_decision_opt_mutators(opt)
        super().__init__(opt, shared)


class TriviaQASearchDecisionTeacher(tqa_sdm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_decision_opt_mutators(opt)
        super().__init__(opt, shared)


class Convai2SearchDecisionTeacher(convai2_sdm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_decision_opt_mutators(opt)
        super().__init__(opt, shared)


class EDSearchDecisionTeacher(ed_sdm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_decision_opt_mutators(opt)
        super().__init__(opt, shared)


class MSCSearchDecisionTeacher(msc_sdm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_decision_opt_mutators(opt)
        super().__init__(opt, shared)


class WoiSearchDecisionTeacher(woi_sdm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_decision_opt_mutators(opt)
        super().__init__(opt, shared)


class WowSearchDecisionTeacher(wow_sdm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_decision_opt_mutators(opt)
        super().__init__(opt, shared)


class Convai2MemoryDecisionTeacher(convai2_mdm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_decision_opt_mutators(opt)
        super().__init__(opt, shared)


class EDMemoryDecisionTeacher(ed_mdm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_decision_opt_mutators(opt)
        super().__init__(opt, shared)


class MSCMemoryDecisionTeacher(msc_mdm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_decision_opt_mutators(opt)
        super().__init__(opt, shared)


class BSTMemoryDecisionTeacher(bst_mdm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_decision_opt_mutators(opt)
        super().__init__(opt, shared)
