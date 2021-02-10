#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDeprecatedDialogTeacher
from .build import build

import copy
import os
from typing import List, Tuple, Optional, TypeVar

from parlai.core.message import Message
from parlai.core.metrics import ExactMatchMetric


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'WebQuestions', dt + '.txt')


class DefaultTeacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt)
        super().__init__(opt, shared)

    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ) -> None:
        if "text" in model_response:
            self.metrics.add(
                "exact_match",
                ExactMatchMetric.compute(
                    guess=model_response.get("text"), answers=labels
                ),
            )
