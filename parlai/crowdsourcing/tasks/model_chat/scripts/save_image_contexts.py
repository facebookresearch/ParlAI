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
from parlai.core.opt import Opt
from parlai.core.worlds import create_task
from parlai.crowdsourcing.tasks.model_chat import run
from parlai.crowdsourcing.tasks.model_chat.utils import (
    get_context_generator,
    get_image_src,
)
from parlai.scripts.display_data import setup_args


def setup_image_context_args():
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
    return task_parser


def save_image_contexts(task_opt: Opt):
    """
    Save a JSON of images and associated contexts for the model image chat task.

    Note that each image will have BST-style context information saved with it, such as
    persona strings and a pair of lines of dialogue from another dataset.
    TODO: perhaps have the image chat task make use of this context information
    """

    print('Creating teacher to loop over images.')
    agent = RepeatLabelAgent(task_opt)
    world = create_task(task_opt, agent)
    num_examples = task_opt['num_examples']

    print('Creating context generator.')
    context_generator = get_context_generator()

    print(f'Looping over {num_examples:d} images and pulling a context for each one.')
    image_contexts = []
    unique_image_srcs = set()
    while len(image_contexts) < num_examples:

        # Get the next teacher act
        world.parley()
        teacher_act = world.get_acts()[0]

        image_src = get_image_src(image=teacher_act['image'])
        if image_src in unique_image_srcs:
            # Skip over non-unique images, such as from the later turns of an episode
            print('\tSkipping non-unique image.')
        else:
            unique_image_srcs.add(image_src)
            image_context = {
                'image_act': teacher_act,
                'context_info': context_generator.get_context(),
            }
            image_contexts.append(image_context)
            if len(image_contexts) % 5 == 0:
                print(f'Collected {len(image_contexts):d} images.')

    print(f'{len(image_contexts):d} image contexts created.')

    # Save
    with open(task_opt['image_context_path'], 'wb') as f:
        pickle.dump(image_contexts, f)


if __name__ == '__main__':
    task_opt_ = setup_image_context_args().parse_args()
    save_image_contexts(task_opt_)
