#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from parlai.core.opt import Opt
from parlai.tasks.fromfile.agents import ParlaiformatTeacher
from parlai.tasks.style_gen.build import (
    build_personality_list,
    build_style_labeled_datasets,
    get_style_labeled_data_folder,
    TASK_FOLDER_NAME,
)


def get_style_labeled_data_path(opt: Opt, base_task: str) -> str:
    """
    Return the filepath for the specified datatype of the specified base task, with an
    Image-Chat personality attached to each example.
    """
    build_style_labeled_datasets(opt)
    # Build the data if it doesn't exist.
    dt = opt['datatype'].split(':')[0]
    return os.path.join(
        get_style_labeled_data_folder(opt['datapath']), base_task, dt + '.txt'
    )


def get_personality_list_path(opt: Opt) -> str:
    """
    Return the path to a list of personalities in the Image-Chat train set.
    """
    build_personality_list(opt)
    # Build the data if it doesn't exist.
    return os.path.join(opt['datapath'], TASK_FOLDER_NAME, 'personality_list.txt')


class LabeledBlendedSkillTalkTeacher(ParlaiformatTeacher):
    """
    Teacher for blended_skill_talk:BlendedSkillTalk, with Image-Chat
    personalities added to examples.
    """

    def __init__(self, opt, shared=None):
        opt['fromfile_datapath'] = get_style_labeled_data_path(
            opt=opt, base_task='blended_skill_talk'
        )
        super().__init__(opt, shared=shared)


class LabeledConvAI2PTTeacher(ParlaiformatTeacher):
    """
    Teacher for blended_skill_talk:ConvAI2PersonaTopicifier, with Image-Chat
    personalities added to examples.
    """

    def __init__(self, opt, shared=None):
        opt['fromfile_datapath'] = get_style_labeled_data_path(
            opt=opt, base_task='blended_skill_talk:ConvAI2PersonaTopicifierTeacher'
        )
        super().__init__(opt, shared=shared)


class LabeledEmpatheticDialoguesPTTeacher(ParlaiformatTeacher):
    """
    Teacher for blended_skill_talk:EDPersonaTopicifier, with Image-Chat
    personalities added to examples.
    """

    def __init__(self, opt, shared=None):
        opt['fromfile_datapath'] = get_style_labeled_data_path(
            opt=opt, base_task='blended_skill_talk:EDPersonaTopicifierTeacher'
        )
        super().__init__(opt, shared=shared)


class LabeledWizardOfWikipediaPTTeacher(ParlaiformatTeacher):
    """
    Teacher for blended_skill_talk:WoWPersonaTopicifier, with Image-Chat
    personalities added to examples.
    """

    def __init__(self, opt, shared=None):
        opt['fromfile_datapath'] = get_style_labeled_data_path(
            opt=opt, base_task='blended_skill_talk:WoWPersonaTopicifierTeacher'
        )
        super().__init__(opt, shared=shared)
