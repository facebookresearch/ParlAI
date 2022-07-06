#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain

from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.params import ParlaiParser
from parlai.core.teachers import Teacher
from parlai.core.worlds import create_task


def get_teacher(config) -> Teacher:
    """
    Return teacher for use in drawing passages for QA.
    """
    parser = ParlaiParser(True, False)
    opt = parser.parse_args(
        list(chain.from_iterable(('--' + k, v) for k, v in config.teacher.items()))
    )
    agent = RepeatLabelAgent(opt)
    return create_task(opt, agent).get_task_agent()
