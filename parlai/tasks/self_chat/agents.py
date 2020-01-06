#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Model self chat.
"""

from parlai.core.agents import create_task_agent_from_taskname
from parlai.core.teachers import Teacher


class DefaultTeacher(Teacher):
    def __init__(self, opt, shared=None):
        raise RuntimeError(
            '-t self_chat is a dummy helper, and not meant to be used directly.'
        )


def create_agents(opt):
    if not opt.get('interactive_task', False):
        return create_task_agent_from_taskname(opt)
    else:
        # interactive task has no task agents (they are attached as user agents)
        return []
