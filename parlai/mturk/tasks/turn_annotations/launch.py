#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from parlai.mturk.tasks.turn_annotations.run import run_task
from parlai.core.params import ParlaiParser


def launch():
    """
    Convenience function to avoid launching from the command line all the time.
    """

    argparser = ParlaiParser(False, False)
    datapath = os.path.join(argparser.parlai_home, 'data')
    task_folder = 'turn_annotations'
    # keys of models_needed_dict will be used as unique names in the output
    # files this says we want 110 conversations from this model.
    models_needed_dict = {'TODO_FIXME_MODEL_UNIQUE_NAME': 110}
    override_opt = {
        'block_qualification': 'block_qualification_name',
        'base_save_folder': os.path.join(datapath, task_folder),
        'onboard_worker_answer_folder': os.path.join(
            datapath, task_folder, 'onboard_answers'
        ),
        'base_model_folder': 'TODO_FIXME_BASE_MODEL_FOLDER',
        'num_conversations': sum(models_needed_dict.values()),
        'is_sandbox': True,
        'reward': 3,
        'num_turns': 6,
        'conversations_needed': models_needed_dict,
    }
    for k, _ in override_opt.items():
        if 'TODO_FIXME' in str(override_opt[k]):
            raise Exception(f'Please customize the option: {k} for this task to run.')
    run_task(override_opt)


if __name__ == '__main__':
    launch()
