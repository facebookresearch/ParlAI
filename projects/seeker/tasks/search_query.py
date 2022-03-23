#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
SeeKeR Search Query Tasks.
"""
import parlai.tasks.wizard_of_internet.agents as woi
import parlai.utils.logging as logging

import projects.seeker.tasks.mutators  # type: ignore


class WoiSearchQueryTeacher(woi.SearchQueryTeacher):
    def __init__(self, opt, shared=None):
        opt['only_last_search_query'] = True
        mutators = '+'.join(
            ['flatten', 'prompt_search_query_mutator', 'skip_retrieval_mutator']
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
        self.id = 'WoiSearchQueryTeacher'


class SearchQueryTeacher(WoiSearchQueryTeacher):
    pass


class DefaultTeacher(SearchQueryTeacher):
    pass
