#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from parlai.core.opt import Opt
from parlai.core.teachers import ParlAIDialogTeacher
from parlai.tasks.style_gen.build import (
    build_personality_list,
    build_style_labeled_datasets,
    get_style_labeled_data_folder,
    TASK_FOLDER_NAME,
)
from parlai.tasks.wrapper.agents import AbstractWrapperTeacher


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


class LabeledBlendedSkillTalkTeacher(ParlAIDialogTeacher):
    """
    Teacher for blended_skill_talk:BlendedSkillTalk, with Image-Chat personalities added
    to examples.
    """

    def __init__(self, opt, shared=None):
        opt['parlaidialogteacher_datafile'] = get_style_labeled_data_path(
            opt=opt, base_task='blended_skill_talk'
        )
        super().__init__(opt, shared=shared)


class LabeledConvAI2PersonaTopicifierTeacher(ParlAIDialogTeacher):
    """
    Teacher for blended_skill_talk:ConvAI2PersonaTopicifier, with Image-Chat
    personalities added to examples.
    """

    def __init__(self, opt, shared=None):
        opt['parlaidialogteacher_datafile'] = get_style_labeled_data_path(
            opt=opt, base_task='blended_skill_talk:ConvAI2PersonaTopicifierTeacher'
        )
        super().__init__(opt, shared=shared)


class LabeledEDPersonaTopicifierTeacher(ParlAIDialogTeacher):
    """
    Teacher for blended_skill_talk:EDPersonaTopicifier, with Image-Chat personalities
    added to examples.
    """

    def __init__(self, opt, shared=None):
        opt['parlaidialogteacher_datafile'] = get_style_labeled_data_path(
            opt=opt, base_task='blended_skill_talk:EDPersonaTopicifierTeacher'
        )
        super().__init__(opt, shared=shared)


class LabeledWoWPersonaTopicifierTeacher(ParlAIDialogTeacher):
    """
    Teacher for blended_skill_talk:WoWPersonaTopicifier, with Image-Chat personalities
    added to examples.
    """

    def __init__(self, opt, shared=None):
        opt['parlaidialogteacher_datafile'] = get_style_labeled_data_path(
            opt=opt, base_task='blended_skill_talk:WoWPersonaTopicifierTeacher'
        )
        super().__init__(opt, shared=shared)


class PrevCurrUttStyleTeacher(AbstractWrapperTeacher):
    # TODO: revise all from here
    """
    Teacher that will shift message['labels'][0] into message['text'] for whatever task
    is specified with --wrapper-task.

    Because the dialogue history is effectively overwritten by this action, all episodes
    will be flattened into one example each.
    """

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)

    def act(self):
        """
        Act on the previous observation.
        """
        act = self.task.act()
        new_act = copy.deepcopy(act)
        if 'labels' in act or 'eval_labels' in act:
            labels_type = 'labels' if 'labels' in act else 'eval_labels'
            labels = act[labels_type]
            if len(labels) != 1:
                raise ValueError('LabelToTextTeacher can only be used with one label!')
            new_act.force_set('text', labels[0])
            new_act.force_set(labels_type, [''])
        else:
            assert 'text' not in act and act['episode_done'] is True
        new_act.force_set('episode_done', True)  # Clear the dialogue history
        return new_act
