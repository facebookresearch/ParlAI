#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Save a pickle of images and associated contexts for the model image chat task.
"""

import os
import pickle

from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.crowdsourcing.tasks.model_chat import run
from parlai.crowdsourcing.tasks.model_chat.utils import get_context_generator
from parlai.scripts.display_data import setup_args


def save_image_contexts():
    """
    Save a JSON of images and associated contexts for the model image chat task.
    """

    print('Creating teacher to loop over images.')
    task_parser = setup_args()
    default_image_context_path = os.path.join(
        os.path.dirname(run.__file__), 'task_config', 'image_contexts'
    )
    task_parser.add_argument(
        '--image-context-path',
        type=str,
        default=default_image_context_path,
        help='Save path for image context file',
    )
    task_opt = task_parser.parse_args()
    agent = RepeatLabelAgent(task_opt)
    world = create_task(task_opt, agent)

    print('Creating context generator.')
    context_generator = get_context_generator()

    print('Looping over images and pulling a context for each one.')
    image_contexts = []
    for _ in range(task_opt['num_examples']):
        world.parley()
        teacher_act = world.get_acts()[0]
        image_context = {
            'image_act': teacher_act,
            'context_info': context_generator.get_context(),
        }
        image_contexts.append(image_context)
    print(f'{len(image_contexts):d} image contexts created.')

    # Save
    with open(task_opt['image_context_path'], 'wb') as f:
        pickle.dump(image_contexts, f)


if __name__ == '__main__':
    save_image_contexts()
