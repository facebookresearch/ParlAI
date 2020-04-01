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
import parlai.tasks.dialogue_nli.agents as dnli
from copy import deepcopy


class AnliTeacher(anli.DefaultTeacher):
    pass


class MultinliTeacher(multinli.DefaultTeacher):
    pass


class SnliTeacher(snli.DefaultTeacher):
    pass


class DnliTeacher(dnli.DefaultTeacher):
    pass


class NliTeacher(MultiTaskTeacher):
    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('NLI Teacher Args')
        parser.add_argument(
            '-dfm',
            '--dialog-format',
            type='bool',
            default=False,
            help="True if one would like to convert to a dialogue format without special tokens such as 'Premise'"
            " and 'Hypothesis' (default: False).",
        )
        parser.add_argument(
            '-bcl',
            '--binary-classes',
            type='bool',
            default=False,
            help="True if label candidates are (contradiction, not_contradiction), and (entailment, contradiction, "
            "neutral) otherwise (default: False).",
        )

    def __init__(self, opt, shared=None):
        nli_tasks = [
            'anli:r1',
            'anli:r2',
            'anli:r3',
            'multinli',
            'snli',
            'dialogue_nli',
        ]
        opt = deepcopy(opt)
        opt['task'] = ','.join(nli_tasks)
        super().__init__(opt, shared)


class DefaultTeacher(NliTeacher):
    pass
