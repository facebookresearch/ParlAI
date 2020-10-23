#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dummy file for imports from all teachers.

Default teacher is a multitask teacher.
"""

from .convai2 import Convai2Teacher  # noqa: F401
from .funpedia import FunpediaTeacher  # noqa: F401
from .light import LIGHTTeacher as LightTeacher  # noqa: F401
from .image_chat import ImageChatTeacher  # noqa: F401
from .opensubtitles import OpensubtitlesTeacher  # noqa: F401
from .wikipedia import WikipediaTeacher  # noqa: F401
from .wizard import WizardTeacher  # noqa: F401
from .yelp import YelpTeacher  # noqa: F401
from .new_data import MdGenderTeacher  # noqa: F401

from parlai.core.teachers import create_task_agent_from_taskname


# task string for multitasking
MULTITASK_STR = (
    'md_gender:convai2,'
    'md_gender:funpedia,'
    'md_gender:image_chat,'
    'md_gender:light,'
    'md_gender:opensubtitles,'
    'md_gender:wikipedia,'
    'md_gender:wizard,'
    'md_gender:yelp'
)

# task string for multi-tasking all tasks for about data
ABOUT_STR = (
    'md_gender:funpedia,'
    'md_gender:image_chat,'
    'md_gender:wikipedia,'
    'md_gender:wizard'
)


class DefaultTeacher:
    def __init__(self, opt, shared):
        raise RuntimeError(
            f'Default teacher does not exist. Please use one of the subteachers, for example: {MULTITASK_STR}'
        )


def create_agents(opt):
    if not opt.get('interactive_task', False):
        return create_task_agent_from_taskname(opt)
    else:
        # interactive task has no task agents (they are attached as user agents)
        return []
