#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from parlai.tasks.sensitive_topics_evaluation.build import build

import json


class SensitiveTopicsEvaluationTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        if 'valid' not in self.datatype:
            raise RuntimeError(
                f'Datatype: {self.datatype} not supported. '
                'This teacher only supports valid.'
            )

        if not shared:
            self.datafile = build(opt)
        else:
            self.datafile = shared['datafile']

        opt['datafile'] = self.datafile
        super().__init__(opt, shared)
        self.id = 'Sensitive Topics Evaluation Topics Valid Teacher'

    def setup_data(self, path):
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                example = json.loads(line)
                del example['episode_done']
                yield example, True

    def share(self):
        shared = super().share()
        shared['datafile'] = self.datafile
        return shared


class DefaultTeacher(SensitiveTopicsEvaluationTeacher):
    pass
