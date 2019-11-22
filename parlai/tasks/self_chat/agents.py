#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Model self chat.
"""

from parlai.core.agents import create_task_agent_from_taskname
from parlai.core.teachers import Teacher
from .build import build

import json
import os
import random


class DefaultTeacher(Teacher):
    # Dummy class.
    pass


def create_agents(opt, task):
    if not opt.get('interactive_task', False):
        return create_task_agent_from_taskname(opt)
    else:
        # interactive task has no task agents (they are attached as user agents)
        return []
