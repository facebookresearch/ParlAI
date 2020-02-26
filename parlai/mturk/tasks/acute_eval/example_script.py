#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.tasks.acute_eval.run import AcuteEvaluator, add_args

"""
Example script for running ACUTE-EVAL.
The only argument that *must* be modified for this to be run is:
``pairings_filepath``:  Path to pairings file in the format specified in the README.md

The following args are useful to tweak to fit your specific needs;
    - ``annotations_per_pair``: A useful arg if you'd like to evaluate a given conversation pair
                                more than once.
    - ``num_matchup_pairs``:    Essentially, how many pairs of conversations you would like to evaluate
    - ``subtasks_per_hit``:     How many comparisons you'd like a turker to complete in one HIT

Help strings for the other arguments can be found in run.py.
"""


def set_args():
    args = add_args()
    # pairings file
    args['pairings_filepath'] = 'parlai/mturk/tasks/acute_eval/example/pairings.jsonl'

    # onboarding and blocking
    args['block_on_onboarding_fail'] = True
    args['block_qualification'] = 'onboarding_qual_name'

    # general ParlAI mturk settings
    args['assignment_duration_in_seconds'] = 600
    args['reward'] = 0.5  # amount to pay workers per hit
    args['max_hits_per_worker'] = 2  # max # hits a worker may complete
    args['is_sandbox'] = True  # set to False to release real hits

    args['annotations_per_pair'] = 1  # num times to use the same conversation pair
    args['num_matchup_pairs'] = 2  # num pairs of conversations to be compared
    args['seed'] = 42  # random seed
    args['subtasks_per_hit'] = 2  # num comparisons to show within one hit

    # question phrasing
    args['s1_choice'] = 'I would prefer to talk to <Speaker 1>'
    args['s2_choice'] = 'I would prefer to talk to <Speaker 2>'
    args['question'] = 'Who would you prefer to talk to for a long conversation?'

    args['num_conversations'] = int(
        args['num_matchup_pairs'] / max((args['subtasks_per_hit'] - 1), 1)
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
    runner = AcuteEvaluator(args)
    runner.run()
