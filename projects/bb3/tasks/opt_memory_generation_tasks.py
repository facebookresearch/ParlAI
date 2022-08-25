#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.opt import Opt
import parlai.utils.logging as logging

from parlai.tasks.msc.agents import PersonaSummaryTeacher
from projects.bb3.tasks.module_level_tasks import BB3TeacherMixin


class MscMemoryGenerationTeacher(BB3TeacherMixin, PersonaSummaryTeacher):
    def __init__(self, opt: Opt, shared=None) -> None:
        opt['include_last_session'] = True
        opt['nopersona_subsampling_weight'] = 0.05
        mutators = '+'.join(
            [
                'skip_retrieval_mutator',
                'prompt_memory_mutator',
                'prefix_speakers_opt',
                'format_gen_tasks_for_decoder_only',
            ]
        )
        if opt.get('mutators'):
            mutators = '+'.join([mutators, opt['mutators']])
        logging.warning(f'overriding mutators to {mutators}')
        opt['mutators'] = mutators
        super().__init__(opt, shared)
