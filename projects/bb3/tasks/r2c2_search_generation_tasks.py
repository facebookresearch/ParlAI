#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import parlai.utils.logging as logging
from projects.seeker.tasks.search_query import WoiSearchQueryTeacher as woi_sgm
from projects.bb3.tasks.module_level_tasks import BB3TeacherMixin
from parlai.tasks.fits.agents import QueryTeacher


class WoiSearchQueryTeacher(BB3TeacherMixin, woi_sgm):
    pass


class FitsSearchQueryTeacher(BB3TeacherMixin, QueryTeacher):
    def __init__(self, opt, shared=None):
        opt['query_source'] = 'human_gold'

        mutators = '+'.join(
            [
                'flatten',
                'prompt_search_query_mutator',
                'skip_retrieval_mutator',
                'fits_pop_keys_mutator',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
