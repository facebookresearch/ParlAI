#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

from parlai.core.teachers import MultiTaskTeacher
import parlai.tasks.anli.agents as anli
import parlai.tasks.multinli.agents as multinli
import parlai.tasks.snli.agents as snli
from copy import deepcopy


class AnliTeacher(anli.DefaultTeacher):
    pass


class MultinliTeacher(multinli.DefaultTeacher):
    pass


class SnliTeacher(snli.DefaultTeacher):
    pass


class NliTeacher(MultiTaskTeacher):
    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('NLI Teacher Args')
        parser.add_argument(
            '--to-parlaitext',
            type='bool',
            default=False,
            help="True if one would like to convert to 'Parlai Text' format (default: False)",
        )
    def __init__(self, opt, shared=None):
        nli_tasks = [
            'anli',
            'multinli',
            'snli',
        ]
        opt = deepcopy(opt)
        opt['task'] = ','.join(nli_tasks)
        super().__init__(opt, shared)


class DefaultTeacher(NliTeacher):
    pass
