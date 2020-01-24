#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.tasks.acute_eval.run import main as run_main, add_args

"""
Example script for running ACUTE-EVAL.
The only arguments that *must* be modified for this to be run are:
'dialogs_path':  Path to folder containing logs of all model conversations.
    Each model should have one file named '<modelname>.jsonl'
'model_comparisons': List of tuples indicating pairs of models to be compared.
    The model names here should match the names of files in the folder
'onboarding_tasks': List of tuples in format (id1, id2, name of matchup) where
    id1, id2 are conversation ids where id2 is the correct conversation's id
Help strings for the other arguments can be found in run.py
"""


def set_args():
    args = add_args()
    # task folder containing pairings file
    args['task_folder'] = 'parlai/mturk/tasks/acute_eval/example/'

    # onboarding amd blocking
    args['block_on_onboarding'] = True
    args['block_qualification'] = 'onboarding_qual_name'

    # general ParlAI mturk settings
    args['assignment_duration_in_seconds'] = 600
    args['reward'] = 0.5  # amount to pay workers per hit
    args['max_hits_per_worker'] = 1  # max # hits a worker may complete
    args['is_sandbox'] = True  # set to False to release real hits

    args['annotations_per_pair'] = 1  # num times to use the same conversation pair
    args['pairs_per_matchup'] = 160  # num pairs of conversations per pair of models
    args['seed'] = 42  # np and torch random seed
    args['subtasks_per_hit'] = 2  # num comparisons to show within one hit

    # question phrasing
    args['s1_choice'] = 'I would prefer to talk to <Speaker 1>'
    args['s2_choice'] = 'I would prefer to talk to <Speaker 2>'
    args['question'] = 'Who would you prefer to talk to for a long conversation?'

    args['num_conversations'] = int(
        len(args['model_comparisons'])
        * args['pairs_per_matchup']
        / (args['task_description']['num_subtasks'] - 1)
    )  # release enough hits to finish all annotations requested

    # Task display on MTurk
    args['task_config'] = {
        'hit_title': 'Which Conversational Partner is Better?',
        'hit_description': 'Evaluate quality of conversations through comparison.',
        'hit_keywords': 'chat,evaluation,comparison,conversation',
    }

    return args


if __name__ == '__main__':
    args = set_args()
    run_main(args)
