#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Save a JSON of image IDs and associated contexts for the model image chat task.
"""

import json
import os

from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from parlai.crowdsourcing.tasks.model_chat import run
from parlai.crowdsourcing.tasks.model_chat.utils import get_context_generator


def save_image_contexts():
    """
    Save a JSON of image IDs and associated contexts for the model image chat task.
    """

    print('Creating teacher to loop over images and personalities.')
    task_parser = ParlaiParser(add_parlai_args=True, add_model_args=False)
    default_image_context_path = os.path.join(
        os.path.dirname(run.__file__), 'task_config', 'image_contexts.json'
    )
    task_parser.add_argument(
        '--image-context-path',
        type=str,
        default=default_image_context_path,
        help='Save path for image context file',
    )
    task_opt = task_parser.parse_args()
    if task_opt['image_context_path'] is None:
        raise ValueError('--image-context-path must be specified!')
    agent = RepeatLabelAgent(task_opt)
    world = create_task(task_opt, agent)

    print('Looping over images and personalities.')
    image_names = []
    personalities = []
    while not world.epoch_done():
        world.parley()
        teacher_act = world.get_acts()[0]
        image_names.append(f"{teacher_act['image_id']:d}.jpg")
        if 'personality' not in teacher_act:
            raise ValueError(
                'The teacher must have an associated personality for each image, as '
                'in the image_chat:Generation task.'
            )
        personalities.append(teacher_act['personality'])
    print(f'{len(image_names):d} sets of images and personalities looped over.')

    print('Picking out a context to use for each image.')
    context_generator = get_context_generator()
    image_names_to_context_info = {
        image_name: context_generator.get_context() for image_name in image_names
    }

    print('For each context, adding in the corresponding personality.')
    for (image_name, context_info), personality in zip(
        image_names_to_context_info.items(), personalities
    ):
        context_info['bot_personality'] = personality

    # Save
    with open(task_opt['image_context_path'], 'w') as f:
        json.dump(image_names_to_context_info, f)


if __name__ == '__main__':
    save_image_contexts()
