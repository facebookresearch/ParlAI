#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.opt import Opt
import parlai.utils.logging as logging

from projects.bb3.tasks.r2c2_search_generation_tasks import (
    FitsSearchQueryTeacher as fits_sgm,
    WoiSearchQueryTeacher as woi_sgm,
)


def get_search_generation_opt_mutators(opt: Opt):
    mutators = '+'.join(['prefix_speakers_opt', 'format_gen_tasks_for_decoder_only'])
    if opt.get('mutators'):
        mutators = '+'.join([opt['mutators'], mutators])
    logging.warning(f'overriding mutators to {mutators}')
    return mutators


class WoiSearchQueryTeacher(woi_sgm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_search_generation_opt_mutators(opt)
        super().__init__(opt, shared)


class FitsSearchQueryTeacher(fits_sgm):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['mutators'] = get_search_generation_opt_mutators(opt)
        super().__init__(opt, shared)
