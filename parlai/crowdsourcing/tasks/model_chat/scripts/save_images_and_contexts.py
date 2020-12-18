#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# TODO: revise below
"""
Select the persona strings and additional context to use when displaying each image.
"""

import json

from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from parlai.tasks.blended_skill_talk.agents import ContextGenerator
from parlai_internal.projects.multimodal_blender.chats_with_images.util import (
    IMAGES_AND_CONTEXTS_PATH,
)


def select_context_per_image():
    """
    Select the persona strings and additional context to use when displaying each image.
    """

    print('Creating teacher to loop over selected images + personalities.')
    task_parser = ParlaiParser(add_parlai_args=True, add_model_args=False)
    task_args = f"""\
    --task internal:multimodal_blender:imageDialogFirstTurnSelectedImages \
    --datatype test \
    """
    task_opt = task_parser.parse_args(task_args.split())
    agent = RepeatLabelAgent(task_opt)
    world = create_task(task_opt, agent)

    print('Looping over selected images + personalities.')
    image_names = []
    personalities = []
    while not world.epoch_done():
        world.parley()
        teacher_act = world.get_acts()[0]
        image_names.append(f"{teacher_act['image_id']:d}.jpg")
        personalities.append(teacher_act['personality'])
    print(f'{len(image_names):d} pairs of images and personalities found.')

    print('Pulling out context to use for each image.')
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    context_opt = argparser.parse_args()
    context_generator = ContextGenerator(context_opt, datatype='test')
    image_names_to_context_info = {
        image_name: context_generator.get_context() for image_name in image_names
    }

    print('For each context, adding in the corresponding personality.')
    for (image_name, context_info), personality in zip(
        image_names_to_context_info.items(), personalities
    ):
        context_info['bot_personality'] = personality

    # Save
    with open(IMAGES_AND_CONTEXTS_PATH, 'w') as f:
        json.dump(image_names_to_context_info, f)


if __name__ == '__main__':
    select_context_per_image()
