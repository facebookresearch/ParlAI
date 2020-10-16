#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.mturk.tasks.turn_annotations.run import run_task


def launch():
    """
    Convenience function to avoid launching from the command line all the time.
    """
    # Keys of models_needed_dict will be used as unique names in the output files. This
    # specifies that we want 110 conversations from this model.
    models_needed_dict = {'TODO_FIXME_MODEL_UNIQUE_NAME': 110}
    # For each key `model_name` in `models_needed_dict`, there should be a model file
    # located at `os.path.join(base_model_folder, model_name, 'model')`
    override_opt = {
        'block_qualification': 'TODO_FIXME_BLOCK_QUALIFICATION_NAME',
        'base_model_folder': 'TODO_FIXME_BASE_MODEL_FOLDER',
        'num_conversations': sum(models_needed_dict.values()),
        'is_sandbox': True,
        'reward': 3,
        'num_turns': 6,
        'conversations_needed': models_needed_dict,
        'task_model_parallel': True,
        'worker_blocklist': [],
        'check_acceptability': False,
        'include_persona': False,
        'conversation_start_mode': 'hi',
        'annotations_intro': 'Does this comment from your partner have any of the following attributes? (Check all that apply)',
        'annotations_config_path': 'TODO_FIXME_ANNOTATIONS_CONFIG_PATH',
        'onboard_task_data_path': 'TODO_FIXME_ONBOARD_TASK_DATA_PATH',
    }
    for k, _ in override_opt.items():
        if 'TODO_FIXME' in str(override_opt[k]):
            raise Exception(f'Please customize the option: {k} for this task to run.')
    run_task(override_opt)


if __name__ == '__main__':
    launch()
